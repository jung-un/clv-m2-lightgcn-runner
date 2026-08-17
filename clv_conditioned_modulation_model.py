"""CLV-conditioned modulation inside a single LightGCN embedding space.

N/V behavioural features do not create independent recommendation scores.
Instead, four low-rank modulators softly rescale the ordinary user/item ID
embeddings before the normal LightGCN propagation.  The output projections are
zero-initialised, so the model starts exactly from an ordinary LightGCN while
remaining trainable by the same plain BPR objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_dual_axis_model import DualItemProfile


class _LowRankModulator(nn.Module):
    """Bias-free feature -> low-rank -> embedding-dimension modulation."""

    def __init__(self, input_dim: int, rank: int, output_dim: int):
        super().__init__()
        if min(input_dim, rank, output_dim) <= 0:
            raise ValueError("modulator dimensions must be positive")
        self.input = nn.Linear(input_dim, rank, bias=False)
        self.output = nn.Linear(rank, output_dim, bias=False)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.input(features)))


class CLVConditionedModulationLightGCN(nn.Module):
    """One-space M2 model with CLV N/V-conditioned ID modulation.

    ``tau`` is a fixed structural bound, not an evaluation-time intervention
    coefficient.  With ``tau=0.10``, every layer-0 dimension is rescaled within
    [0.9, 1.1].  There is no independent N/V dot product and no post-score
    residual.
    """

    EVAL_AXIS_MODES = frozenset({"both", "n_only", "v_only", "none"})

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_activity: np.ndarray,
        user_value: np.ndarray,
        user_activity_valid: np.ndarray,
        user_value_valid: np.ndarray,
        item_profile: DualItemProfile,
        adj: torch.Tensor,
        embedding_dim: int = 64,
        modulation_rank: int = 4,
        tau: float = 0.10,
        n_layers: int = 3,
        pref_reg: float = 1e-4,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be positive")
        if not 0 < tau <= 0.5:
            raise ValueError("tau must be in (0, 0.5]")
        if n_layers < 0:
            raise ValueError("n_layers must be non-negative")

        user_activity = self._feature_matrix(
            user_activity, n_users, "user_activity"
        )
        user_value = self._feature_matrix(user_value, n_users, "user_value")
        item_activity = self._feature_matrix(
            item_profile.activity, n_items, "item_activity"
        )
        item_value = self._feature_matrix(
            item_profile.value, n_items, "item_value"
        )
        user_activity_valid = self._valid_mask(
            user_activity_valid, n_users, "user_activity_valid"
        )
        user_value_valid = self._valid_mask(
            user_value_valid, n_users, "user_value_valid"
        )
        item_valid = self._valid_mask(
            item_profile.valid_item, n_items, "item_valid"
        )
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape does not match user/item counts")

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.tau = float(tau)
        self.n_layers = n_layers
        self.pref_reg = float(pref_reg)
        self.eval_axis_mode = "both"

        # Create ID embeddings first so a shared seed reproduces ordinary M1
        # initialisation before any M2-only module consumes RNG state.
        self.E_u = nn.Embedding(n_users, embedding_dim)
        self.E_i = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.user_n = _LowRankModulator(
            user_activity.shape[1], modulation_rank, embedding_dim
        )
        self.user_v = _LowRankModulator(
            user_value.shape[1], modulation_rank, embedding_dim
        )
        self.item_n = _LowRankModulator(
            item_activity.shape[1], modulation_rank, embedding_dim
        )
        self.item_v = _LowRankModulator(
            item_value.shape[1], modulation_rank, embedding_dim
        )

        self.register_buffer("user_activity", torch.from_numpy(user_activity))
        self.register_buffer("user_value", torch.from_numpy(user_value))
        self.register_buffer("item_activity", torch.from_numpy(item_activity))
        self.register_buffer("item_value", torch.from_numpy(item_value))
        self.register_buffer(
            "user_activity_valid",
            torch.from_numpy(user_activity_valid.astype(np.float32)),
        )
        self.register_buffer(
            "user_value_valid",
            torch.from_numpy(user_value_valid.astype(np.float32)),
        )
        self.register_buffer(
            "item_valid", torch.from_numpy(item_valid.astype(np.float32))
        )
        self.register_buffer("adj", adj.coalesce())

    @staticmethod
    def _feature_matrix(values, expected_rows: int, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != expected_rows:
            raise ValueError(f"{name} must have shape ({expected_rows}, d)")
        if array.shape[1] == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite and non-empty")
        return array

    @staticmethod
    def _valid_mask(values, expected_rows: int, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=bool)
        if array.shape != (expected_rows,):
            raise ValueError(f"{name} must have shape ({expected_rows},)")
        return array

    def set_eval_axes(self, mode: str) -> None:
        if mode not in self.EVAL_AXIS_MODES:
            raise ValueError(f"unsupported eval axis mode: {mode}")
        self.eval_axis_mode = mode

    def _axis_modulations(self):
        user_n = self.user_n(self.user_activity) * self.user_activity_valid[:, None]
        user_v = self.user_v(self.user_value) * self.user_value_valid[:, None]
        item_n = self.item_n(self.item_activity) * self.item_valid[:, None]
        item_v = self.item_v(self.item_value) * self.item_valid[:, None]
        return user_n, user_v, item_n, item_v

    def _combined_modulations(self):
        user_n, user_v, item_n, item_v = self._axis_modulations()
        if self.eval_axis_mode == "both":
            return user_n + user_v, item_n + item_v
        if self.eval_axis_mode == "n_only":
            return user_n, item_n
        if self.eval_axis_mode == "v_only":
            return user_v, item_v
        return torch.zeros_like(user_n), torch.zeros_like(item_n)

    def layer0_embeddings(self):
        user_modulation, item_modulation = self._combined_modulations()
        user_scale = 1.0 + self.tau * torch.tanh(user_modulation)
        item_scale = 1.0 + self.tau * torch.tanh(item_modulation)
        return self.E_u.weight * user_scale, self.E_i.weight * item_scale

    def propagate(self):
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        """Compatibility with the common evaluator; scoring uses one dot only."""
        user, item = self.propagate()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, w=None):
        if w is not None:
            raise ValueError("M4 sample weights cannot be used in the M2 model")
        if float(lam) != 0.0:
            raise ValueError("M2 modulation has no external lambda score residual")
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
    def modulation_diagnostics(self) -> dict[str, float]:
        user_n, user_v, item_n, item_v = self._axis_modulations()
        combined = torch.cat(
            [torch.tanh(user_n + user_v), torch.tanh(item_n + item_v)], dim=0
        )
        return {
            "tau": self.tau,
            "user_n_abs_mean": float(user_n.abs().mean()),
            "user_v_abs_mean": float(user_v.abs().mean()),
            "item_n_abs_mean": float(item_n.abs().mean()),
            "item_v_abs_mean": float(item_v.abs().mean()),
            "combined_saturation_share": float(combined.abs().gt(0.95).float().mean()),
        }
