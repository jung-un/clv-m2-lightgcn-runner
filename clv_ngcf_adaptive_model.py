"""NGCF propagation with the frozen adaptive M2 applied after layer concatenation.

Paper-style explicit self message: LeakyReLU(W1(h + A h) + W2(h * A h)).
As in the authors' implementation, apply message dropout, append normalized
layer outputs, and pass the unnormalized dropped output to the next layer.
This is a protocol-matched NGCF implementation, not an exact reproduction of
the authors' TensorFlow benchmark: symmetric loop-free adjacency plus explicit
self path, normal(.1) ID initialization and project ID/shared L2 are recorded.
No node dropout: the frozen M3 graph must not be randomly resampled.
"""
import torch
from torch import nn
from torch.nn import functional as F

from clv_dataset_adaptive_orthogonal_nv_model import DatasetAdaptiveOrthogonalNVLightGCN


class AdaptiveNVNGCF(DatasetAdaptiveOrthogonalNVLightGCN):
    """Same backbone parameters for all arms; only M2 arms retain feature encoders."""

    def __init__(self, *, m2_active, message_dropout=.1, backbone_l2=1e-6, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= message_dropout < 1 or backbone_l2 < 0:
            raise ValueError("NGCF dropout/L2 범위 오류")
        self.m2_active = bool(m2_active)
        self.message_dropout = float(message_dropout)
        self.backbone_l2 = float(backbone_l2)
        self.output_dim = self.id_dim * (self.n_layers + 1)
        self.sum_layers = nn.ModuleList(nn.Linear(self.id_dim, self.id_dim) for _ in range(self.n_layers))
        self.bi_layers = nn.ModuleList(nn.Linear(self.id_dim, self.id_dim) for _ in range(self.n_layers))
        for layer in [*self.sum_layers, *self.bi_layers]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Construct the backbone before branching so the same seed gives exactly
        # the same ID/NGCF initialization in all eight arms.
        if self.m2_active:
            self.encoders = nn.ModuleDict({name: nn.Linear(3, self.output_dim, bias=False)
                                          for name in self.encoders})
            for layer in self.encoders.values():
                nn.init.normal_(layer.weight, std=.02)
        else:
            del self.encoders, self.axis_logits, self.rho_logit
            for name in list(self._buffers):
                if name != "adj":
                    delattr(self, name)

    def _id_embeddings(self):
        current = torch.cat((self.E_u.weight, self.E_i.weight))
        layers = [current]
        for linear, interaction in zip(self.sum_layers, self.bi_layers):
            neighbour = torch.sparse.mm(self.adj, current)
            current = F.leaky_relu(linear(current + neighbour)
                                   + interaction(current * neighbour), negative_slope=.2)
            current = F.dropout(current, p=self.message_dropout, training=self.training)
            layers.append(F.normalize(current, dim=1, eps=1e-12))
        final = torch.cat(layers, dim=1)
        return final[:self.n_users], final[self.n_users:]

    def embeddings(self, need_value=True):
        if self.m2_active:
            return super().embeddings(need_value=need_value)
        users, items = self._id_embeddings()
        return users, items, users.new_zeros((self.n_users, 1)), items.new_zeros((self.n_items, 1))

    def _pair_scores(self, users, positives, negatives):
        if self.m2_active:
            # Keeps positive-item leave-one-user-out features, negative-item
            # full profiles, the two N/V reliability factors and adaptive mix.
            return super()._pair_scores(users, positives, negatives)
        user, item = self._id_embeddings()
        return (user[users] * item[positives]).sum(1), (user[users] * item[negatives]).sum(1)

    def batch_l2(self, users, positives, negatives):
        if self.m2_active:
            penalty = super().batch_l2(users, positives, negatives)
        else:
            penalty = self.pref_reg * sum(x.square().sum() for x in
                (self.E_u(users), self.E_i(positives), self.E_i(negatives))) / len(users)
        return penalty + self.backbone_l2 * sum(p.square().sum() for layer in
            (self.sum_layers, self.bi_layers) for p in layer.parameters())

    def training_gradient_diagnostics(self):
        result = super().training_gradient_diagnostics() if self.m2_active else {}
        for name, parameter in self.named_parameters():
            if name.startswith(("E_", "sum_layers", "bi_layers")):
                result[name + "_gradient_norm"] = (float(parameter.grad.norm())
                    if parameter.grad is not None else None)
        return result

    @torch.no_grad()
    def representation_diagnostics(self):
        result = super().representation_diagnostics() if self.m2_active else {}
        return result | {"backbone": "NGCF", "m2_active": self.m2_active,
            "id_dim": self.id_dim, "output_dim": self.output_dim,
            "message_dropout": self.message_dropout, "node_dropout": 0.,
            "backbone_l2": self.backbone_l2,
            "parameter_count": sum(p.numel() for p in self.parameters())}
