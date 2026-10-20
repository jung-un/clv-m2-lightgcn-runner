"""Two-layer GCN encoder for BPR recommendation with the fixed M2 intervention.

S = D^-1/2 (A_w + I) D^-1/2, Z = S ReLU(S E W0 + b0) W1 + b1.
Self-loop weight is 1 for every arm. M3 changes only observed edge weights
before normalization. ID inputs and BPR decoder replace the original GCN
node-classification inputs/head; this is not that benchmark's reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F

from clv_dataset_adaptive_orthogonal_nv_model import DatasetAdaptiveOrthogonalNVLightGCN


class AdaptiveNVGCN(DatasetAdaptiveOrthogonalNVLightGCN):
    def __init__(self, *, m2_active, backbone_l2=1e-6, **kwargs):
        super().__init__(**kwargs)
        if self.n_layers != 2 or backbone_l2 < 0:
            raise ValueError("GCN은 사전 고정한 2층 / 비음수 L2를 사용합니다")
        self.m2_active = bool(m2_active)
        self.backbone_l2 = float(backbone_l2)
        self.output_dim = self.id_dim
        self.gcn_layers = nn.ModuleList(
            nn.Linear(self.id_dim, self.id_dim, bias=True) for _ in range(self.n_layers))
        for layer in self.gcn_layers:
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
        for index, layer in enumerate(self.gcn_layers):
            # Bias is added AFTER aggregation (S H W + b, not S(H W + b)).
            current = layer(torch.sparse.mm(self.adj, current))
            if index < self.n_layers - 1:
                current = F.relu(current)
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
        return penalty + self.backbone_l2 * sum(p.square().sum() for p in self.gcn_layers.parameters())

    def training_gradient_diagnostics(self):
        result = super().training_gradient_diagnostics() if self.m2_active else {}
        for name, parameter in self.named_parameters():
            if name.startswith(("E_", "gcn_layers")):
                result[name + "_gradient_norm"] = (
                    float(parameter.grad.norm()) if parameter.grad is not None else None)
        return result

    @torch.no_grad()
    def representation_diagnostics(self):
        result = super().representation_diagnostics() if self.m2_active else {}
        result.update(backbone="GCN", m2_active=self.m2_active, id_dim=self.id_dim,
            output_dim=self.output_dim, self_loops=True, raw_self_loop_weight=1.,
            feature_dropout=0., backbone_l2=self.backbone_l2,
            activation="ReLU hidden; linear output", layer_aggregation="final layer only",
            normalized_weight_min=float(self.adj.values().min()),
            normalized_weight_max=float(self.adj.values().max()),
            parameter_count=sum(p.numel() for p in self.parameters()))
        return result
