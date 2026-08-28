"""Recent-level and within-user-change conditioning for one LightGCN.

At each chronological training context, two fixed historical conditions are
available for every user: the recent CLV-proxy level and its change from the
preceding window.  Both condition the same 64-dimensional user-ID embedding
inside the LightGCN forward graph.  There is no item feature, score residual,
graph weight, sample weight, or additional loss term.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCLVLevelChangeLightGCN(nn.Module):
    """A single LightGCN shared by chronological CLV contexts.

    For context ``k`` and user ``u``::

        scale = 1 + rho * tanh(level * w_level + change * w_change)
        E_user = NormPreserve(E_user_id * scale)

    Level and change are fixed, train-history-only inputs in [-1, 1].  The two
    dimension weights and the ID embeddings are optimized jointly by the same
    plain pairwise BPR objective.  Setting ``rho=0`` gives the exact matched
    multi-anchor LightGCN control.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        adjacencies: Sequence[torch.Tensor],
        level_conditions: Sequence[np.ndarray | torch.Tensor],
        change_conditions: Sequence[np.ndarray | torch.Tensor],
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
            len(adjacencies)
            == len(level_conditions)
            == len(change_conditions)
            == len(context_names)
        ):
            raise ValueError("context inputs must have equal length")
        if not context_names or len(set(context_names)) != len(context_names):
            raise ValueError("context_names must be non-empty and unique")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.context_names = tuple(str(name) for name in context_names)
        self._active_context = 0

        # ID tables are constructed first so rho=0 and rho>0 arms consume the
        # same random numbers for their shared parameters.
        self.E_u = nn.Embedding(n_users, embedding_dim)
        self.E_i = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.level_dimension_weight = nn.Parameter(torch.zeros(embedding_dim))
        self.change_dimension_weight = nn.Parameter(torch.zeros(embedding_dim))
        if self.rho == 0.0:
            self.level_dimension_weight.requires_grad_(False)
            self.change_dimension_weight.requires_grad_(False)

        for index, (adj, level, change) in enumerate(
            zip(
                adjacencies,
                level_conditions,
                change_conditions,
                strict=True,
            )
        ):
            if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
                raise ValueError(f"adjacency {index} has the wrong shape")
            level_tensor = self._validated_condition(level, n_users, "level", index)
            change_tensor = self._validated_condition(
                change, n_users, "change", index
            )
            self.register_buffer(f"_adj_{index}", adj.coalesce(), persistent=False)
            self.register_buffer(
                f"_level_{index}", level_tensor, persistent=False
            )
            self.register_buffer(
                f"_change_{index}", change_tensor, persistent=False
            )

    @staticmethod
    def _validated_condition(value, n_users: int, name: str, index: int):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape != (n_users,):
            raise ValueError(
                f"{name} condition {index} must have shape ({n_users},)"
            )
        if not torch.isfinite(tensor).all() or tensor.abs().max() > 1.000001:
            raise ValueError(f"{name} conditions must be finite and in [-1, 1]")
        return tensor

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
        index = self._active_context
        return (
            getattr(self, f"_adj_{index}"),
            getattr(self, f"_level_{index}"),
            getattr(self, f"_change_{index}"),
        )

    @staticmethod
    def _preserve_row_norm(original, transformed):
        original_norm = original.norm(dim=1, keepdim=True)
        transformed_norm = transformed.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return transformed * (original_norm / transformed_norm)

    def layer0_embeddings(self):
        _, level, change = self._context_tensors()
        combined = (
            level[:, None] * self.level_dimension_weight[None, :]
            + change[:, None] * self.change_dimension_weight[None, :]
        )
        scale = 1.0 + self.rho * torch.tanh(combined)
        transformed = self._preserve_row_norm(self.E_u.weight, self.E_u.weight * scale)
        return transformed, self.E_i.weight

    def propagate(self):
        adj, _, _ = self._context_tensors()
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        # ``need_value`` is retained for the common evaluator interface.
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
        levels = torch.stack(
            [getattr(self, f"_level_{i}") for i in range(len(self.context_names))]
        )
        changes = torch.stack(
            [getattr(self, f"_change_{i}") for i in range(len(self.context_names))]
        )
        combined = (
            levels[:, :, None] * self.level_dimension_weight[None, None, :]
            + changes[:, :, None] * self.change_dimension_weight[None, None, :]
        )
        scales = 1.0 + self.rho * torch.tanh(combined)
        level_weight = torch.tanh(self.level_dimension_weight)
        change_weight = torch.tanh(self.change_dimension_weight)
        return {
            "rho": self.rho,
            "level_dimension_weight_abs_mean": float(level_weight.abs().mean()),
            "level_dimension_weight_abs_max": float(level_weight.abs().max()),
            "change_dimension_weight_abs_mean": float(change_weight.abs().mean()),
            "change_dimension_weight_abs_max": float(change_weight.abs().max()),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "level_across_anchor_std_mean": float(levels.std(dim=0).mean()),
            "change_across_anchor_std_mean": float(changes.std(dim=0).mean()),
            "level_changing_user_share": float(
                levels.std(dim=0).gt(1e-8).float().mean()
            ),
            "change_changing_user_share": float(
                changes.std(dim=0).gt(1e-8).float().mean()
            ),
        }
