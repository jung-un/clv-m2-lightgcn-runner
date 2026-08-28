"""Time-indexed historical-CLV conditioning for a single LightGCN.

The model keeps the ordinary 64-dimensional user/item ID embeddings.  At each
training anchor, a user's *historical* CLV proxy changes, and that scalar
conditions a bounded feature-wise transformation of the same user embedding.
There is no separate CLV score, item-economic feature, graph weight, or loss
weight.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCLVModulationLightGCN(nn.Module):
    """One LightGCN shared by several chronological training anchors.

    For context ``k`` and user ``u``::

        scale[u, k] = 1 + rho * c[u, k] * tanh(axis_weight)
        E_user[u, k] = NormPreserve(E_user_id[u] * scale[u, k])

    ``c[u, k]`` is fixed historical information in [-1, 1].  ``axis_weight``
    is learned by the same plain BPR loss as the ID embeddings.  ``rho=0`` is
    the matched multi-anchor M1 control and is numerically identical to an
    unconditioned LightGCN under the same graph/label schedule.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        adjacencies: Sequence[torch.Tensor],
        clv_conditions: Sequence[np.ndarray | torch.Tensor],
        context_names: Sequence[str],
        embedding_dim: int = 64,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be positive")
        if embedding_dim <= 0 or n_layers < 0:
            raise ValueError("invalid embedding_dim or n_layers")
        if not 0.0 <= rho <= 0.25:
            raise ValueError("rho must be in [0, 0.25]")
        if not (
            len(adjacencies) == len(clv_conditions) == len(context_names)
        ):
            raise ValueError("context inputs must have equal length")
        if not context_names:
            raise ValueError("at least one context is required")
        if len(set(context_names)) != len(context_names):
            raise ValueError("context_names must be unique")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.context_names = tuple(str(name) for name in context_names)
        self._active_context = 0

        # ID tables are created first so M1 and M2 share the same seed-wise
        # initialisation.  The CLV-only parameter is zero initialised.
        self.E_u = nn.Embedding(n_users, embedding_dim)
        self.E_i = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.clv_dimension_weight = nn.Parameter(torch.zeros(embedding_dim))
        if self.rho == 0.0:
            self.clv_dimension_weight.requires_grad_(False)

        for index, (adj, condition) in enumerate(
            zip(adjacencies, clv_conditions, strict=True)
        ):
            if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
                raise ValueError(f"adjacency {index} has the wrong shape")
            value = torch.as_tensor(condition, dtype=torch.float32)
            if value.shape != (n_users,):
                raise ValueError(f"CLV condition {index} must have shape ({n_users},)")
            if not torch.isfinite(value).all() or value.abs().max() > 1.000001:
                raise ValueError("CLV conditions must be finite and in [-1, 1]")
            # These tensors are fully reproducible from the train history and
            # should not inflate checkpoints.
            self.register_buffer(f"_adj_{index}", adj.coalesce(), persistent=False)
            self.register_buffer(f"_condition_{index}", value, persistent=False)

    @property
    def active_context_name(self) -> str:
        return self.context_names[self._active_context]

    def set_context(self, context: int | str) -> None:
        if isinstance(context, str):
            try:
                index = self.context_names.index(context)
            except ValueError as exc:
                raise ValueError(f"unknown context: {context}") from exc
        else:
            index = int(context)
        if index < 0 or index >= len(self.context_names):
            raise IndexError("context index out of range")
        self._active_context = index

    def _context_tensors(self):
        return (
            getattr(self, f"_adj_{self._active_context}"),
            getattr(self, f"_condition_{self._active_context}"),
        )

    @staticmethod
    def _preserve_row_norm(
        original: torch.Tensor, transformed: torch.Tensor
    ) -> torch.Tensor:
        original_norm = original.norm(dim=1, keepdim=True)
        transformed_norm = transformed.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return transformed * (original_norm / transformed_norm)

    def layer0_embeddings(self):
        _, condition = self._context_tensors()
        direction = torch.tanh(self.clv_dimension_weight)
        scale = 1.0 + self.rho * condition[:, None] * direction[None, :]
        transformed_user = self.E_u.weight * scale
        transformed_user = self._preserve_row_norm(
            self.E_u.weight, transformed_user
        )
        return transformed_user, self.E_i.weight

    def propagate(self):
        adj, _ = self._context_tensors()
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        """Compatibility with the common evaluator; only one dot score exists."""
        user, item = self.propagate()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, w=None):
        if w is not None:
            raise ValueError("M4 sample weights cannot be used in M2")
        if float(lam) != 0.0:
            raise ValueError("M2 has no external score residual")
        user, item = self.propagate()
        positive_score = (user[users] * item[positives]).sum(1)
        negative_score = (user[users] * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def condition_diagnostics(self) -> dict[str, float]:
        direction = torch.tanh(self.clv_dimension_weight)
        conditions = torch.stack(
            [getattr(self, f"_condition_{i}") for i in range(len(self.context_names))]
        )
        scales = 1.0 + self.rho * conditions[:, :, None] * direction[None, None, :]
        return {
            "rho": self.rho,
            "clv_dimension_weight_abs_mean": float(direction.abs().mean()),
            "clv_dimension_weight_abs_max": float(direction.abs().max()),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "condition_across_anchor_std_mean": float(conditions.std(dim=0).mean()),
            "condition_changing_user_share": float(
                conditions.std(dim=0).gt(1e-8).float().mean()
            ),
        }
