"""Jointly trained affine N/V history expressions; no external reranking."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from clv_history_item_fit_model import HistoryItemFitLightGCN


class LinearNVHistoryLightGCN(HistoryItemFitLightGCN):
    """Replace q*p by W[p;q]+b, retaining the existing ID and item paths.

    Constant-q controls retain N/V-derived histories and their validity masks;
    they are NOT a control removing all historical economic information.
    """

    def __init__(self, *, constant_q=False, **kwargs):
        for key in ("q_n", "q_v"):
            if not np.isfinite(kwargs[key]).all():
                raise ValueError(f"{key} must be finite after invalid-user masking")
        super().__init__(**kwargs)
        self.constant_q = bool(constant_q)
        # Paired actual/control initialization; do not alter the base RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.n_transform = nn.Linear(self.axis_dim + 1, self.axis_dim)
            self.v_transform = nn.Linear(self.axis_dim + 1, self.axis_dim)
        self.register_buffer("n_has_history", torch.sparse.sum(self.activity_history, 1).to_dense() > 1e-8)
        self.register_buffer("v_has_history", torch.sparse.sum(self.value_history, 1).to_dense() > 1e-8)

    def _transform(self, profile, q, valid, has_history, layer):
        if self.constant_q:
            q = torch.full_like(q, 0.5)
        transformed = layer(torch.cat([profile, q[:, None]], dim=1))
        return transformed * (valid.bool() & has_history)[:, None]

    def _full_history_profiles(self):
        sn, tn, sv, tv = self._axis_tables()
        pn = torch.sparse.mm(self.activity_history, sn)
        pv = torch.sparse.mm(self.value_history, sv)
        return (
            self._transform(pn, self.q_n, self.activity_valid, self.n_has_history, self.n_transform), tn,
            self._transform(pv, self.q_v, self.value_valid, self.v_has_history, self.v_transform), tv,
        )

    def _training_profiles(self, users, positives):
        sn, tn, sv, tv = self._axis_tables()
        an, av = self._positive_history_shares(users, positives)
        pn = self._leave_one_out(torch.sparse.mm(self.activity_history, sn)[users], sn[positives], an)
        pv = self._leave_one_out(torch.sparse.mm(self.value_history, sv)[users], sv[positives], av)
        return (
            self._transform(pn, self.q_n[users], self.activity_valid[users],
                            self.n_has_history[users] & ((1-an) > 1e-8), self.n_transform), tn,
            self._transform(pv, self.q_v[users], self.value_valid[users],
                            self.v_has_history[users] & ((1-av) > 1e-8), self.v_transform), tv,
        )

    def batch_l2(self, users, positives, negatives, need_value=False):
        # Same existing L2 coefficient, extended to shared affine parameters.
        # Shared params counted once per batch, not once per sampled user.
        shared = sum(p.square().sum() for layer in (self.n_transform, self.v_transform)
                     for p in layer.parameters())
        return super().batch_l2(users, positives, negatives, need_value) + self.pref_reg * shared

    @torch.no_grad()
    def representation_diagnostics(self):
        result = super().representation_diagnostics()
        result.update(constant_q=self.constant_q, affine_parameter_count=sum(
            p.numel() for layer in (self.n_transform, self.v_transform) for p in layer.parameters()))
        for name, layer in (("n", self.n_transform), ("v", self.v_transform)):
            result[f"{name}_q_column_norm"] = float(layer.weight[:, -1].norm())
            result[f"{name}_history_matrix_norm"] = float(layer.weight[:, :-1].norm())
            result[f"{name}_bias_norm"] = float(layer.bias.norm())
        return result

    def training_gradient_diagnostics(self):
        result = super().training_gradient_diagnostics()
        for name, layer in (("n", self.n_transform), ("v", self.v_transform)):
            for field, parameter in layer.named_parameters():
                result[f"{name}_{field}_gradient_norm"] = (
                    0.0 if parameter.grad is None else float(parameter.grad.detach().norm()))
        return result
