"""N/V-specific personal-history to candidate-item fit inside one recommender.

The ID block is a standard LightGCN representation.  The two small N/V blocks
are learned from a user's own item history rather than from an item-global
repeat-share or popularity statistic.  Source-item and target-item embeddings
are intentionally separate, following the source/target factorization used by
item-similarity models such as FISM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PersonalHistoryWeights:
    users: np.ndarray
    items: np.ndarray
    activity_share: np.ndarray
    value_share: np.ndarray
    diagnostics: dict[str, float]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    value = pd.Series(left).rank(method="average").corr(
        pd.Series(right).rank(method="average")
    )
    return 0.0 if pd.isna(value) else float(value)


def build_personal_history_weights(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
) -> PersonalHistoryWeights:
    """Build train-only, within-user N/V item-history shares.

    N share is the number of distinct baskets containing an item divided by
    the sum of those counts over the user's history.  V share is the item's
    non-negative purchase amount divided by the user's total purchase amount.
    Neither value contains a global item-repeat or item-popularity feature.
    """
    required = {"u_idx", "i_idx", "b_raw", "v"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"개인 구매이력 가중치 입력 열 누락: {sorted(missing)}")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users와 n_items는 양수여야 합니다")

    frame = train.loc[:, ["u_idx", "i_idx", "b_raw", "v"]].copy()
    frame["positive_value"] = frame["v"].clip(lower=0.0)
    pairs = (
        frame.groupby(["u_idx", "i_idx"], sort=True)
        .agg(
            basket_count=("b_raw", "nunique"),
            purchase_amount=("positive_value", "sum"),
        )
        .reset_index()
    )
    users = pairs["u_idx"].to_numpy(np.int64)
    items = pairs["i_idx"].to_numpy(np.int64)
    if ((users < 0) | (users >= n_users)).any():
        raise ValueError("u_idx가 사용자 범위를 벗어났습니다")
    if ((items < 0) | (items >= n_items)).any():
        raise ValueError("i_idx가 아이템 범위를 벗어났습니다")

    activity_mass = pairs["basket_count"].to_numpy(np.float64)
    value_mass = pairs["purchase_amount"].to_numpy(np.float64)
    activity_total = np.bincount(
        users, weights=activity_mass, minlength=n_users
    ).astype(np.float64)
    value_total = np.bincount(
        users, weights=value_mass, minlength=n_users
    ).astype(np.float64)
    activity_share = np.divide(
        activity_mass,
        activity_total[users],
        out=np.zeros_like(activity_mass),
        where=activity_total[users] > 0,
    )
    value_share = np.divide(
        value_mass,
        value_total[users],
        out=np.zeros_like(value_mass),
        where=value_total[users] > 0,
    )

    activity_row_sum = np.bincount(
        users, weights=activity_share, minlength=n_users
    )
    value_row_sum = np.bincount(users, weights=value_share, minlength=n_users)
    activity_valid = activity_total > 0
    value_valid = value_total > 0
    degree = np.bincount(items, minlength=n_items).astype(np.float64)
    received_n = np.bincount(
        items, weights=activity_share, minlength=n_items
    ).astype(np.float64)
    received_v = np.bincount(
        items, weights=value_share, minlength=n_items
    ).astype(np.float64)
    observed_items = degree > 0
    diagnostics = {
        "history_edge_count": int(len(pairs)),
        "activity_valid_user_share": float(activity_valid.mean()),
        "value_valid_user_share": float(value_valid.mean()),
        "activity_row_sum_max_error": float(
            np.abs(activity_row_sum[activity_valid] - 1.0).max()
            if activity_valid.any()
            else 0.0
        ),
        "value_row_sum_max_error": float(
            np.abs(value_row_sum[value_valid] - 1.0).max()
            if value_valid.any()
            else 0.0
        ),
        "activity_received_mass_degree_spearman": _spearman(
            received_n[observed_items], degree[observed_items]
        ),
        "value_received_mass_degree_spearman": _spearman(
            received_v[observed_items], degree[observed_items]
        ),
        "activity_share_mean": float(activity_share.mean()),
        "value_share_mean": float(value_share.mean()),
    }
    return PersonalHistoryWeights(
        users=users,
        items=items,
        activity_share=activity_share.astype(np.float32),
        value_share=value_share.astype(np.float32),
        diagnostics=diagnostics,
    )


class HistoryItemFitLightGCN(nn.Module):
    """LightGCN ID preference plus two direct history-candidate fit blocks."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        history: PersonalHistoryWeights,
        q_n: np.ndarray,
        q_v: np.ndarray,
        activity_valid: np.ndarray,
        value_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        axis_dim: int = 4,
        n_layers: int = 2,
        rho: float = 0.05,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0 or id_dim <= 0 or axis_dim <= 0:
            raise ValueError("사용자·아이템·임베딩 차원은 양수여야 합니다")
        if n_layers < 0:
            raise ValueError("n_layers는 0 이상이어야 합니다")
        if not 0.0 < rho <= 0.1:
            raise ValueError("rho는 0보다 크고 0.1 이하여야 합니다")
        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        valid_n = np.asarray(activity_valid, dtype=bool)
        valid_v = np.asarray(value_valid, dtype=bool)
        if not (q_n.shape == q_v.shape == valid_n.shape == valid_v.shape == (n_users,)):
            raise ValueError("사용자 N/V 수준과 유효성 shape이 다릅니다")
        if ((q_n < 0) | (q_n > 1) | (q_v < 0) | (q_v > 1)).any():
            raise ValueError("q_N과 q_V는 0과 1 사이여야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.axis_dim = int(axis_dim)
        self.n_layers = int(n_layers)
        self.rho = float(rho)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        self.activity_source = nn.Embedding(n_items, axis_dim)
        self.activity_target = nn.Embedding(n_items, axis_dim)
        self.value_source = nn.Embedding(n_items, axis_dim)
        self.value_target = nn.Embedding(n_items, axis_dim)
        for table in (
            self.E_u,
            self.E_i,
            self.activity_source,
            self.activity_target,
            self.value_source,
            self.value_target,
        ):
            nn.init.normal_(table.weight, std=0.1)

        users = np.asarray(history.users, dtype=np.int64)
        items = np.asarray(history.items, dtype=np.int64)
        indices = torch.from_numpy(np.stack([users, items]))
        shape = (n_users, n_items)
        activity_matrix = torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(np.asarray(history.activity_share, np.float32)),
            shape,
        ).coalesce()
        value_matrix = torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(np.asarray(history.value_share, np.float32)),
            shape,
        ).coalesce()
        keys = users * np.int64(n_items) + items
        order = np.argsort(keys, kind="stable")
        self.register_buffer("activity_history", activity_matrix)
        self.register_buffer("value_history", value_matrix)
        self.register_buffer("history_keys", torch.from_numpy(keys[order]))
        self.register_buffer(
            "history_activity_share",
            torch.from_numpy(np.asarray(history.activity_share, np.float32)[order]),
        )
        self.register_buffer(
            "history_value_share",
            torch.from_numpy(np.asarray(history.value_share, np.float32)[order]),
        )
        self.register_buffer("q_n", torch.from_numpy(q_n * valid_n))
        self.register_buffer("q_v", torch.from_numpy(q_v * valid_v))
        self.register_buffer("activity_valid", torch.from_numpy(valid_n.astype(np.float32)))
        self.register_buffer("value_valid", torch.from_numpy(valid_v.astype(np.float32)))
        self.register_buffer("adj", adj.coalesce())
        self.history_diagnostics = dict(history.diagnostics)

    @property
    def total_dim(self) -> int:
        return self.id_dim + 2 * self.axis_dim

    @staticmethod
    def _unit_rows(values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values, dim=1, eps=1e-8)

    def _id_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def _axis_tables(self):
        return (
            self._unit_rows(self.activity_source.weight),
            self._unit_rows(self.activity_target.weight),
            self._unit_rows(self.value_source.weight),
            self._unit_rows(self.value_target.weight),
        )

    def _full_history_profiles(self):
        source_n, target_n, source_v, target_v = self._axis_tables()
        profile_n = torch.sparse.mm(self.activity_history, source_n)
        profile_v = torch.sparse.mm(self.value_history, source_v)
        profile_n = profile_n * self.q_n[:, None] * self.activity_valid[:, None]
        profile_v = profile_v * self.q_v[:, None] * self.value_valid[:, None]
        return profile_n, target_n, profile_v, target_v

    def _positive_history_shares(
        self, users: torch.Tensor, positives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = users * self.n_items + positives
        positions = torch.searchsorted(self.history_keys, keys)
        clipped = positions.clamp(max=len(self.history_keys) - 1)
        if not torch.equal(self.history_keys[clipped], keys):
            raise RuntimeError("학습 positive가 개인 구매이력 edge에서 발견되지 않았습니다")
        return (
            self.history_activity_share[clipped],
            self.history_value_share[clipped],
        )

    @staticmethod
    def _leave_one_out(
        full_profile: torch.Tensor,
        source_item: torch.Tensor,
        share: torch.Tensor,
    ) -> torch.Tensor:
        remaining = 1.0 - share
        numerator = full_profile - share[:, None] * source_item
        return torch.where(
            (remaining > 1e-8)[:, None],
            numerator / remaining.clamp_min(1e-8)[:, None],
            torch.zeros_like(numerator),
        )

    def _training_profiles(
        self, users: torch.Tensor, positives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_n, target_n, source_v, target_v = self._axis_tables()
        raw_n = torch.sparse.mm(self.activity_history, source_n)[users]
        raw_v = torch.sparse.mm(self.value_history, source_v)[users]
        share_n, share_v = self._positive_history_shares(users, positives)
        profile_n = self._leave_one_out(raw_n, source_n[positives], share_n)
        profile_v = self._leave_one_out(raw_v, source_v[positives], share_v)
        profile_n = profile_n * self.q_n[users, None] * self.activity_valid[users, None]
        profile_v = profile_v * self.q_v[users, None] * self.value_valid[users, None]
        return profile_n, target_n, profile_v, target_v

    def embeddings(self, need_value: bool = True):
        user_id, item_id = self._id_embeddings()
        user_n, item_n, user_v, item_v = self._full_history_profiles()
        scale = float(np.sqrt(self.rho))
        user = torch.cat([user_id, scale * user_n, scale * user_v], dim=1)
        item = torch.cat([item_id, scale * item_n, scale * item_v], dim=1)
        return (
            user,
            item,
            user.new_zeros((self.n_users, 1)),
            item.new_zeros((self.n_items, 1)),
        )

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        tables = (
            self.E_u.weight[users],
            self.E_i.weight[positives],
            self.E_i.weight[negatives],
            self.activity_source.weight[positives],
            self.activity_target.weight[positives],
            self.activity_target.weight[negatives],
            self.value_source.weight[positives],
            self.value_target.weight[positives],
            self.value_target.weight[negatives],
        )
        return self.pref_reg * sum(table.pow(2).sum() for table in tables) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        user_id, item_id = self._id_embeddings()
        user_n, target_n, user_v, target_v = self._training_profiles(
            users, positives
        )
        positive_score = (user_id[users] * item_id[positives]).sum(1)
        negative_score = (user_id[users] * item_id[negatives]).sum(1)
        positive_score = positive_score + self.rho * (
            (user_n * target_n[positives]).sum(1)
            + (user_v * target_v[positives]).sum(1)
        )
        negative_score = negative_score + self.rho * (
            (user_n * target_n[negatives]).sum(1)
            + (user_v * target_v[negatives]).sum(1)
        )
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        return loss, {
            "bpr": float(bpr.detach()),
            "p_correct": float((positive_score > negative_score).float().mean().detach()),
            "objective": "plain_bpr",
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float]:
        user_n, item_n, user_v, item_v = self._full_history_profiles()
        return {
            "rho": self.rho,
            "total_dim": self.total_dim,
            "q_n_mean": float(self.q_n.mean()),
            "q_n_std": float(self.q_n.std()),
            "q_v_mean": float(self.q_v.mean()),
            "q_v_std": float(self.q_v.std()),
            "activity_history_profile_mean_norm": float(user_n.norm(dim=1).mean()),
            "value_history_profile_mean_norm": float(user_v.norm(dim=1).mean()),
            "activity_target_mean_norm": float(item_n.norm(dim=1).mean()),
            "value_target_mean_norm": float(item_v.norm(dim=1).mean()),
            "activity_source_target_cosine_mean": float(
                (self._unit_rows(self.activity_source.weight) * item_n).sum(1).mean()
            ),
            "value_source_target_cosine_mean": float(
                (self._unit_rows(self.value_source.weight) * item_v).sum(1).mean()
            ),
        }

    def epoch_training_diagnostics(self) -> dict[str, float]:
        return self.representation_diagnostics()

    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            gradient = parameter.grad
            return 0.0 if gradient is None else float(gradient.norm().detach())

        return {
            "activity_source_gradient_norm": norm(self.activity_source.weight),
            "activity_target_gradient_norm": norm(self.activity_target.weight),
            "value_source_gradient_norm": norm(self.value_source.weight),
            "value_target_gradient_norm": norm(self.value_target.weight),
        }
