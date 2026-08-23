"""Low-dimensional N/V representation inside one LightGCN.

The model keeps the ordinary ID preference block, adds two small user-specific
historical-CLV blocks, and learns the corresponding item response from item ID.
It deliberately has no explicit item popularity/price input, user gate, or
learned global axis weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _UserAxisEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(values))


def _center_and_bound(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Centre each coordinate over valid train users and keep it in [-1, 1]."""
    mask = valid[:, None]
    count = mask.sum().clamp_min(1.0)
    centered = (values - (values * mask).sum(0, keepdim=True) / count) * mask
    scale = centered.abs().amax(0, keepdim=True).clamp_min(1.0)
    return centered / scale


class GateFreeLowDimNVLightGCN(nn.Module):
    """ID(64) + activity(4) + transaction-value(4) joint representation."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_activity: np.ndarray,
        user_value: np.ndarray,
        user_activity_valid: np.ndarray,
        user_value_valid: np.ndarray,
        q_n: np.ndarray,
        q_v: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        axis_dim: int = 4,
        hidden_dim: int = 8,
        n_layers: int = 2,
        axis_budget: float = 0.1,
        training_axis_balance_delta: float = 0.0,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0 or id_dim <= 0:
            raise ValueError("사용자·아이템·ID 차원은 양수여야 합니다")
        if axis_dim < 2:
            raise ValueError("축 차원은 수준 1차원과 행동표현을 위해 2 이상이어야 합니다")
        if hidden_dim <= 0 or n_layers < 0:
            raise ValueError("hidden_dim은 양수, n_layers는 0 이상이어야 합니다")
        if not 0.0 < axis_budget <= 1.0:
            raise ValueError("axis_budget은 0보다 크고 1 이하여야 합니다")
        if not 0.0 <= training_axis_balance_delta < 1.0:
            raise ValueError(
                "training_axis_balance_delta는 0 이상 1 미만이어야 합니다"
            )

        user_activity = np.asarray(user_activity, dtype=np.float32)
        user_value = np.asarray(user_value, dtype=np.float32)
        activity_valid = np.asarray(user_activity_valid, dtype=bool)
        value_valid = np.asarray(user_value_valid, dtype=bool)
        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        if user_activity.shape[0] != n_users or user_value.shape[0] != n_users:
            raise ValueError("사용자 특징 행 수가 n_users와 다릅니다")
        expected = (n_users,)
        if any(array.shape != expected for array in (activity_valid, value_valid, q_n, q_v)):
            raise ValueError("사용자 축 수준·유효성 shape이 n_users와 다릅니다")
        if not all(np.isfinite(array).all() for array in (user_activity, user_value, q_n, q_v)):
            raise ValueError("N/V 입력은 모두 유한해야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.axis_dim = int(axis_dim)
        self.n_layers = int(n_layers)
        self.axis_budget = float(axis_budget)
        self.training_axis_balance_delta = float(training_axis_balance_delta)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        residual_dim = axis_dim - 1
        self.activity_user = _UserAxisEncoder(
            user_activity.shape[1], hidden_dim, residual_dim
        )
        self.value_user = _UserAxisEncoder(
            user_value.shape[1], hidden_dim, residual_dim
        )
        self.activity_item = nn.Linear(id_dim, axis_dim, bias=False)
        self.value_item = nn.Linear(id_dim, axis_dim, bias=False)
        nn.init.normal_(self.activity_item.weight, std=0.02)
        nn.init.normal_(self.value_item.weight, std=0.02)

        self.register_buffer("user_activity", torch.from_numpy(user_activity.copy()))
        self.register_buffer("user_value", torch.from_numpy(user_value.copy()))
        self.register_buffer(
            "user_activity_valid", torch.from_numpy(activity_valid.astype(np.float32))
        )
        self.register_buffer(
            "user_value_valid", torch.from_numpy(value_valid.astype(np.float32))
        )
        self.register_buffer("q_n", torch.from_numpy(q_n.copy()))
        self.register_buffer("q_v", torch.from_numpy(q_v.copy()))
        self.register_buffer("adj", adj.coalesce())

    @property
    def total_dim(self) -> int:
        return self.id_dim + 2 * self.axis_dim

    def _user_axis(
        self,
        inputs: torch.Tensor,
        level: torch.Tensor,
        valid: torch.Tensor,
        encoder: _UserAxisEncoder,
    ) -> torch.Tensor:
        mask = valid[:, None]
        valid_count = valid.sum().clamp_min(1.0)
        centered_level = (
            level - (level * valid).sum() / valid_count
        )[:, None]
        raw = torch.cat([centered_level, encoder(inputs)], dim=1) * mask
        return _center_and_bound(raw, valid)

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_n = self._user_axis(
            self.user_activity,
            self.q_n,
            self.user_activity_valid,
            self.activity_user,
        )
        user_v = self._user_axis(
            self.user_value,
            self.q_v,
            self.user_value_valid,
            self.value_user,
        )
        # Item responses are learned from item identity by the same BPR loss;
        # no RepeatShare, degree, price, category, or transaction-value input.
        item_n = torch.tanh(self.activity_item(self.E_i.weight))
        item_v = torch.tanh(self.value_item(self.E_i.weight))
        scale = float(np.sqrt(self.axis_budget))
        user = torch.cat([self.E_u.weight, scale * user_n, scale * user_v], dim=1)
        item = torch.cat([self.E_i.weight, scale * item_n, scale * item_v], dim=1)
        return user, item

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
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

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        user, item = self.propagate()
        id_end = self.id_dim
        activity_end = id_end + self.axis_dim

        def components(item_ids):
            selected_user = user[users]
            selected_item = item[item_ids]
            return (
                (selected_user[:, :id_end] * selected_item[:, :id_end]).sum(1),
                (
                    selected_user[:, id_end:activity_end]
                    * selected_item[:, id_end:activity_end]
                ).sum(1),
                (
                    selected_user[:, activity_end:]
                    * selected_item[:, activity_end:]
                ).sum(1),
            )

        positive_id, positive_activity, positive_value = components(positives)
        negative_id, negative_activity, negative_value = components(negatives)
        if self.training and self.training_axis_balance_delta > 0:
            epsilon = (
                torch.rand_like(positive_id) * 2.0 - 1.0
            ) * self.training_axis_balance_delta
        else:
            epsilon = torch.zeros_like(positive_id)
        activity_multiplier = 1.0 + epsilon
        value_multiplier = 1.0 - epsilon
        # The same epsilon is used for the positive and negative item of each
        # BPR triplet.  Only the relative N/V balance changes; the total axis
        # budget remains constant and both axes stay strictly positive.
        positive_score = (
            positive_id
            + activity_multiplier * positive_activity
            + value_multiplier * positive_value
        )
        negative_score = (
            negative_id
            + activity_multiplier * negative_activity
            + value_multiplier * negative_value
        )
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "objective": "plain_bpr",
                "training_axis_balance_delta": self.training_axis_balance_delta,
                "axis_balance_epsilon_mean": float(epsilon.mean()),
                "axis_balance_epsilon_std": float(
                    epsilon.std(unbiased=False)
                ),
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def representation_diagnostics(self) -> dict:
        user0, item0 = self.layer0_embeddings()
        return {
            "axis_budget": self.axis_budget,
            "training_axis_balance_delta": self.training_axis_balance_delta,
            "training_activity_multiplier_range": [
                1.0 - self.training_axis_balance_delta,
                1.0 + self.training_axis_balance_delta,
            ],
            "training_value_multiplier_range": [
                1.0 - self.training_axis_balance_delta,
                1.0 + self.training_axis_balance_delta,
            ],
            "evaluation_activity_multiplier": 1.0,
            "evaluation_value_multiplier": 1.0,
            "total_dim": self.total_dim,
            "activity_user_coordinate_mean_abs": float(
                user0[:, self.id_dim : self.id_dim + self.axis_dim]
                .mean(0)
                .abs()
                .max()
            ),
            "value_user_coordinate_mean_abs": float(
                user0[:, self.id_dim + self.axis_dim :]
                .mean(0)
                .abs()
                .max()
            ),
            "mean_user_norm": float(user0.norm(dim=1).mean()),
            "mean_item_norm": float(item0.norm(dim=1).mean()),
        }
