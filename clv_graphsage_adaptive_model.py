"""Mean GraphSAGE with fixed M2: self/neighbor concatenation, ELU, L2 per layer.
Project recommendation adaptation: trainable IDs, BPR, mean of layers0/1/2.
Full neighbors; no sampling/dropout/self-loop edges. Not an inductive benchmark.
"""
import torch
from torch import nn
from torch.nn import functional as F

from clv_dataset_adaptive_orthogonal_nv_model import DatasetAdaptiveOrthogonalNVLightGCN


class AdaptiveNVGraphSAGE(DatasetAdaptiveOrthogonalNVLightGCN):
    def __init__(self, *, m2_active, backbone_l2=1e-6, **kwargs):
        super().__init__(**kwargs)
        if self.n_layers != 2 or backbone_l2 < 0:
            raise ValueError("GraphSAGE은 사전 고정한 2층 / 비음수 L2를 사용합니다")
        self.m2_active = bool(m2_active)
        self.backbone_l2 = float(backbone_l2)
        self.output_dim = self.id_dim
        self.graphsage_layers = nn.ModuleList(
            nn.Linear(2 * self.id_dim, self.id_dim, bias=True) for _ in range(self.n_layers))
        for layer in self.graphsage_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Initialize the common backbone before removing unused M2 parameters.
        if not self.m2_active:
            del self.encoders, self.axis_logits, self.rho_logit
            for name in list(self._buffers):
                if name != "adj":
                    delattr(self, name)

    def _id_embeddings(self):
        current = torch.cat((self.E_u.weight, self.E_i.weight))
        layers = [current]
        for layer in self.graphsage_layers:
            neighbors = torch.sparse.mm(self.adj, current)
            current = F.normalize(F.elu(layer(torch.cat((current, neighbors), dim=1))),
                                  p=2, dim=1, eps=1e-12)
            layers.append(current)
        current = torch.stack(layers, dim=0).mean(dim=0)
        return current[:self.n_users], current[self.n_users:]

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
        return penalty + self.backbone_l2 * sum(p.square().sum() for p in self.graphsage_layers.parameters())

    def training_gradient_diagnostics(self):
        result = super().training_gradient_diagnostics() if self.m2_active else {}
        for name, parameter in self.named_parameters():
            if name.startswith(("E_", "graphsage_layers")):
                result[name + "_gradient_norm"] = (
                    float(parameter.grad.norm()) if parameter.grad is not None else None)
        return result

    @torch.no_grad()
    def representation_diagnostics(self):
        result = super().representation_diagnostics() if self.m2_active else {}
        result.update(backbone="GraphSAGE", m2_active=self.m2_active, id_dim=self.id_dim,
            output_dim=self.output_dim, self_loops=False, self_path="concatenation",
            feature_dropout=0., backbone_l2=self.backbone_l2,
            activation="ELU then L2 each layer", layer_aggregation="mean of layers0,1,2",
            normalized_weight_min=float(self.adj.values().min()),
            normalized_weight_max=float(self.adj.values().max()),
            parameter_count=sum(p.numel() for p in self.parameters()))
        return result
