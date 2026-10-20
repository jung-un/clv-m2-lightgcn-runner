"""Single-head GAT recommendation with fixed M2 and M4, and M3 attention priors.

M3: softmax(LeakyReLU(attention logits) + log(raw transaction weight)).
This equals alpha*w/sum(alpha*w), not LightGCN weighted-degree normalization.
Keep binary interaction support, no self loops, and mean(layer0, layer1, layer2),
as in the project's earlier GAT recommendation implementation. The original
GAT node-classification architecture/benchmark is not reproduced exactly.
"""
import torch
from torch import nn
from torch.nn import functional as F

from clv_dataset_adaptive_orthogonal_nv_model import DatasetAdaptiveOrthogonalNVLightGCN
from gat_clv_level_composition_price_model import SparseSingleHeadGATLayer, _weighted_neighbor_sum


class PriorGATLayer(SparseSingleHeadGATLayer):
    def attention_weights(self, features, receiver, source, n_nodes, log_prior):
        transformed = self.projection(features)
        logits = F.leaky_relu((transformed @ self.receiver_attention)[receiver]
                             + (transformed @ self.source_attention)[source], .2)
        logits = logits + log_prior
        maximum = logits.new_full((n_nodes,), -torch.inf)
        # The row shift cancels in softmax; detaching avoids max-tie gradients.
        maximum.scatter_reduce_(0, receiver, logits.detach(), reduce="amax", include_self=True)
        stable = (logits - maximum[receiver]).exp()
        total = stable.new_zeros(n_nodes).scatter_add_(0, receiver, stable)
        weights = stable / total[receiver].clamp_min(1e-12)
        return weights, transformed

    def forward(self, features, receiver, source, n_nodes, log_prior):
        weights, transformed = self.attention_weights(features, receiver, source, n_nodes, log_prior)
        return F.elu(_weighted_neighbor_sum(transformed, receiver, source, weights, n_nodes))


class AdaptiveNVGAT(DatasetAdaptiveOrthogonalNVLightGCN):
    def __init__(self, *, m2_active, feature_dropout=0., backbone_l2=1e-6, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= feature_dropout < 1 or backbone_l2 < 0:
            raise ValueError("GAT dropout/L2 범위 오류")
        self.m2_active = bool(m2_active)
        self.feature_dropout = float(feature_dropout)
        self.backbone_l2 = float(backbone_l2)
        self.output_dim = self.id_dim
        graph = self.adj.coalesce()
        if (not torch.isfinite(graph.values()).all() or (graph.values() <= 0).any()
                or (graph.indices()[0] == graph.indices()[1]).any()):
            raise ValueError("GAT는 자기연결 없는 양수 사전가중치가 필요합니다")
        self.register_buffer("edge_receiver", graph.indices()[0], persistent=False)
        self.register_buffer("edge_source", graph.indices()[1], persistent=False)
        self.register_buffer("edge_log_prior", graph.values().log(), persistent=False)
        self.gat_layers = nn.ModuleList(PriorGATLayer(self.id_dim) for _ in range(self.n_layers))
        # Create the common backbone before branching, so all arms initialize alike.
        if not self.m2_active:
            del self.encoders, self.axis_logits, self.rho_logit
            for name in list(self._buffers):
                if name not in {"adj", "edge_receiver", "edge_source", "edge_log_prior"}:
                    delattr(self, name)

    def _id_embeddings(self):
        current = torch.cat((self.E_u.weight, self.E_i.weight))
        total = current
        for layer in self.gat_layers:
            current = F.dropout(current, self.feature_dropout, self.training)
            current = layer(current, self.edge_receiver, self.edge_source,
                            self.n_users + self.n_items, self.edge_log_prior)
            total = total + current
        final = total / (self.n_layers + 1)
        return final[:self.n_users], final[self.n_users:]

    def embeddings(self, need_value=True):
        if self.m2_active:
            return super().embeddings(need_value=need_value)
        users, items = self._id_embeddings()
        return users, items, users.new_zeros((self.n_users, 1)), items.new_zeros((self.n_items, 1))

    def _pair_scores(self, users, positives, negatives):
        if self.m2_active:
            return super()._pair_scores(users, positives, negatives)
        user, item = self._id_embeddings()
        return (user[users] * item[positives]).sum(1), (user[users] * item[negatives]).sum(1)

    def batch_l2(self, users, positives, negatives):
        if self.m2_active:
            penalty = super().batch_l2(users, positives, negatives)
        else:
            penalty = self.pref_reg * sum(x.square().sum() for x in
                (self.E_u(users), self.E_i(positives), self.E_i(negatives))) / len(users)
        return penalty + self.backbone_l2 * sum(p.square().sum() for p in self.gat_layers.parameters())

    def training_gradient_diagnostics(self):
        result = super().training_gradient_diagnostics() if self.m2_active else {}
        for name, parameter in self.named_parameters():
            if name.startswith(("E_", "gat_layers")):
                result[name + "_gradient_norm"] = (float(parameter.grad.norm())
                    if parameter.grad is not None else None)
        return result

    @torch.no_grad()
    def representation_diagnostics(self):
        result = super().representation_diagnostics() if self.m2_active else {}
        result.update(backbone="GAT", m2_active=self.m2_active, id_dim=self.id_dim,
            output_dim=self.output_dim, attention_heads=1, self_loops=False,
            feature_dropout=self.feature_dropout, attention_dropout=0.,
            backbone_l2=self.backbone_l2, layer_aggregation="mean(layer0,layer1,layer2)",
            prior_min=float(self.edge_log_prior.exp().min()),
            prior_max=float(self.edge_log_prior.exp().max()),
            parameter_count=sum(p.numel() for p in self.parameters()))
        current = torch.cat((self.E_u.weight, self.E_i.weight))
        n = len(current)
        degree = torch.bincount(self.edge_receiver, minlength=n)
        for i, layer in enumerate(self.gat_layers, 1):
            weights, transformed = layer.attention_weights(current, self.edge_receiver,
                self.edge_source, n, self.edge_log_prior)
            row_sum = weights.new_zeros(n).scatter_add_(0, self.edge_receiver, weights)
            entropy = weights.new_zeros(n).scatter_add_(0, self.edge_receiver,
                -weights * weights.clamp_min(1e-12).log())
            active, multi = degree > 0, degree > 1
            result[f"gat_layer{i}_row_sum_max_error"] = float((row_sum[active] - 1).abs().max())
            result[f"gat_layer{i}_normalized_entropy_mean"] = (
                float((entropy[multi] / degree[multi].float().log()).mean()) if multi.any() else 0.)
            current = F.elu(_weighted_neighbor_sum(transformed, self.edge_receiver,
                self.edge_source, weights, n))
        return result
