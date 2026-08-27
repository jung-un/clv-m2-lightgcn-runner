"""Fixed-budget N/V blocks inside one LightGCN.

The module keeps the original ID block and adds two small CLV-related blocks.
The total N/V intervention budget is fixed, while each user's activity/value
composition allocates that budget.  Item-side axis inputs describe the N/V
tendency of an item's historical buyers after removing the expected effect of
item popularity (and, for V, category).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ItemAxisAffinity:
    activity: np.ndarray
    value: np.ndarray
    activity_valid: np.ndarray
    value_valid: np.ndarray
    diagnostics: dict[str, float]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    value = pd.Series(left).rank(method="average").corr(
        pd.Series(right).rank(method="average")
    )
    return 0.0 if pd.isna(value) else float(value)


def _standardize_residual(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if not valid.any():
        return result
    selected = np.asarray(values[valid], dtype=np.float64)
    scale = float(selected.std())
    if scale <= 1e-12:
        return result
    result[valid] = ((selected - selected.mean()) / scale).astype(np.float32)
    return result


def _global_degree_residual(
    buyer_mean: np.ndarray, log_degree: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    residual = np.zeros_like(buyer_mean, dtype=np.float64)
    if valid.sum() < 2:
        return residual
    x = log_degree[valid]
    y = buyer_mean[valid]
    design = np.column_stack([np.ones_like(x), x])
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    residual[valid] = y - design @ coefficient
    return residual


def _category_degree_residual(
    buyer_mean: np.ndarray,
    log_degree: np.ndarray,
    category: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Residual from category fixed effects plus a shared log-degree slope."""
    residual = np.zeros_like(buyer_mean, dtype=np.float64)
    if valid.sum() < 2:
        return residual
    frame = pd.DataFrame(
        {
            "item": np.flatnonzero(valid),
            "y": buyer_mean[valid],
            "x": log_degree[valid],
            "category": category[valid],
        }
    )
    means = frame.groupby("category", sort=False)[["x", "y"]].transform("mean")
    x_within = frame["x"].to_numpy() - means["x"].to_numpy()
    y_within = frame["y"].to_numpy() - means["y"].to_numpy()
    denominator = float(np.dot(x_within, x_within))
    slope = 0.0 if denominator <= 1e-12 else float(
        np.dot(x_within, y_within) / denominator
    )
    expected = means["y"].to_numpy() + slope * x_within
    residual[frame["item"].to_numpy(np.int64)] = frame["y"].to_numpy() - expected
    return residual


def build_popularity_controlled_item_affinities(
    train: pd.DataFrame,
    *,
    n_items: int,
    q_n: np.ndarray,
    q_v: np.ndarray,
    user_activity_valid: np.ndarray,
    user_value_valid: np.ndarray,
) -> ItemAxisAffinity:
    """Build train-only item N/V buyer affinities.

    Each user contributes at most once per item.  N removes the expected buyer
    activity level given log item degree.  V additionally removes category
    fixed effects, so neither axis can use raw item popularity as its input.
    """
    required = {"u_idx", "i_idx", "cat_idx"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"아이템 N/V affinity 입력 열 누락: {sorted(missing)}")
    q_n = np.asarray(q_n, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)
    activity_valid_user = np.asarray(user_activity_valid, dtype=bool)
    value_valid_user = np.asarray(user_value_valid, dtype=bool)
    if not (
        q_n.shape == q_v.shape == activity_valid_user.shape == value_valid_user.shape
    ):
        raise ValueError("사용자 q_N/q_V와 유효성 shape이 다릅니다")

    pairs = train.loc[:, ["u_idx", "i_idx"]].drop_duplicates()
    users = pairs["u_idx"].to_numpy(np.int64)
    items = pairs["i_idx"].to_numpy(np.int64)
    degree = np.bincount(items, minlength=n_items).astype(np.float64)
    log_degree = np.log1p(degree)

    def buyer_mean(level: np.ndarray, valid_user: np.ndarray):
        keep = valid_user[users]
        count = np.bincount(items[keep], minlength=n_items).astype(np.float64)
        total = np.bincount(
            items[keep], weights=level[users[keep]], minlength=n_items
        ).astype(np.float64)
        mean = np.zeros(n_items, dtype=np.float64)
        valid_item = count > 0
        mean[valid_item] = total[valid_item] / count[valid_item]
        return mean, valid_item

    mean_n, valid_n = buyer_mean(q_n, activity_valid_user)
    mean_v, valid_v = buyer_mean(q_v, value_valid_user)
    category = np.full(n_items, -1, dtype=np.int64)
    category_mode = train.groupby("i_idx", sort=False)["cat_idx"].agg(
        lambda values: values.mode().iat[0]
    )
    category[category_mode.index.to_numpy(np.int64)] = category_mode.to_numpy(np.int64)

    raw_n_residual = _global_degree_residual(mean_n, log_degree, valid_n)
    raw_v_residual = _category_degree_residual(
        mean_v, log_degree, category, valid_v
    )
    activity = _standardize_residual(raw_n_residual, valid_n)[:, None]
    value = _standardize_residual(raw_v_residual, valid_v)[:, None]

    diagnostics = {
        "item_activity_valid_share": float(valid_n.mean()),
        "item_value_valid_share": float(valid_v.mean()),
        "item_degree_mean": float(degree.mean()),
        "item_degree_median": float(np.median(degree)),
        "activity_buyer_mean_degree_spearman": _spearman(
            mean_n[valid_n], degree[valid_n]
        ),
        "activity_affinity_degree_spearman": _spearman(
            activity[valid_n, 0], degree[valid_n]
        ),
        "value_buyer_mean_degree_spearman": _spearman(
            mean_v[valid_v], degree[valid_v]
        ),
        "value_affinity_degree_spearman": _spearman(
            value[valid_v, 0], degree[valid_v]
        ),
        "activity_affinity_std": float(activity[valid_n, 0].std()) if valid_n.any() else 0.0,
        "value_affinity_std": float(value[valid_v, 0].std()) if valid_v.any() else 0.0,
    }
    return ItemAxisAffinity(
        activity=activity.astype(np.float32),
        value=value.astype(np.float32),
        activity_valid=valid_n,
        value_valid=valid_v,
        diagnostics=diagnostics,
    )


def fixed_axis_composition(
    q_n: np.ndarray,
    q_v: np.ndarray,
    activity_valid: np.ndarray,
    value_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a fixed user N/V budget without a learned gate."""
    q_n = np.asarray(q_n, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)
    valid_n = np.asarray(activity_valid, dtype=bool)
    valid_v = np.asarray(value_valid, dtype=bool)
    if not (q_n.shape == q_v.shape == valid_n.shape == valid_v.shape):
        raise ValueError("q_N/q_V와 축 유효성 shape이 다릅니다")
    if not np.isfinite(q_n).all() or not np.isfinite(q_v).all():
        raise ValueError("q_N/q_V는 유한해야 합니다")
    weight_n = np.exp(q_n - np.maximum(q_n, q_v)) * valid_n
    weight_v = np.exp(q_v - np.maximum(q_n, q_v)) * valid_v
    denominator = weight_n + weight_v
    pi_n = np.divide(
        weight_n, denominator, out=np.zeros_like(weight_n), where=denominator > 0
    )
    pi_v = np.divide(
        weight_v, denominator, out=np.zeros_like(weight_v), where=denominator > 0
    )
    return pi_n.astype(np.float32), pi_v.astype(np.float32)


class _NormalizedAxisEncoder(nn.Module):
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
        return F.normalize(self.net(values), dim=1, eps=1e-8)


class FixedCompositionNVLightGCN(nn.Module):
    """ID(64)|N(4)|V(4) representation with fixed N/V composition."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_activity: np.ndarray,
        user_value: np.ndarray,
        user_activity_valid: np.ndarray,
        user_value_valid: np.ndarray,
        item_affinity: ItemAxisAffinity,
        q_n: np.ndarray,
        q_v: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        axis_dim: int = 4,
        hidden_dim: int = 8,
        n_layers: int = 2,
        rho: float = 0.05,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0 or id_dim <= 0 or axis_dim <= 0:
            raise ValueError("사용자·아이템·임베딩 차원은 양수여야 합니다")
        if hidden_dim <= 0 or n_layers < 0:
            raise ValueError("hidden_dim은 양수, n_layers는 0 이상이어야 합니다")
        if not 0.0 < rho <= 0.1:
            raise ValueError("rho는 0보다 크고 0.1 이하여야 합니다")

        user_activity = np.asarray(user_activity, dtype=np.float32)
        user_value = np.asarray(user_value, dtype=np.float32)
        activity_valid = np.asarray(user_activity_valid, dtype=bool)
        value_valid = np.asarray(user_value_valid, dtype=bool)
        if user_activity.shape[0] != n_users or user_value.shape[0] != n_users:
            raise ValueError("사용자 축 입력 행 수가 n_users와 다릅니다")
        if item_affinity.activity.shape != (n_items, 1) or item_affinity.value.shape != (n_items, 1):
            raise ValueError("아이템 affinity는 축별 [n_items, 1]이어야 합니다")
        pi_n, pi_v = fixed_axis_composition(q_n, q_v, activity_valid, value_valid)

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.axis_dim = int(axis_dim)
        self.n_layers = int(n_layers)
        self.rho = float(rho)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.activity_user = _NormalizedAxisEncoder(
            user_activity.shape[1], hidden_dim, axis_dim
        )
        self.value_user = _NormalizedAxisEncoder(
            user_value.shape[1], hidden_dim, axis_dim
        )
        self.activity_item = _NormalizedAxisEncoder(1, hidden_dim, axis_dim)
        self.value_item = _NormalizedAxisEncoder(1, hidden_dim, axis_dim)

        self.register_buffer("user_activity", torch.from_numpy(user_activity.copy()))
        self.register_buffer("user_value", torch.from_numpy(user_value.copy()))
        self.register_buffer(
            "user_activity_valid", torch.from_numpy(activity_valid.astype(np.float32))
        )
        self.register_buffer(
            "user_value_valid", torch.from_numpy(value_valid.astype(np.float32))
        )
        self.register_buffer(
            "item_activity", torch.from_numpy(item_affinity.activity.copy())
        )
        self.register_buffer("item_value", torch.from_numpy(item_affinity.value.copy()))
        self.register_buffer(
            "item_activity_valid",
            torch.from_numpy(item_affinity.activity_valid.astype(np.float32)),
        )
        self.register_buffer(
            "item_value_valid",
            torch.from_numpy(item_affinity.value_valid.astype(np.float32)),
        )
        self.register_buffer("pi_n", torch.from_numpy(pi_n))
        self.register_buffer("pi_v", torch.from_numpy(pi_v))
        self.register_buffer("adj", adj.coalesce())

    @property
    def total_dim(self) -> int:
        return self.id_dim + 2 * self.axis_dim

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_n = self.activity_user(self.user_activity) * self.user_activity_valid[:, None]
        user_v = self.value_user(self.user_value) * self.user_value_valid[:, None]
        item_n = self.activity_item(self.item_activity) * self.item_activity_valid[:, None]
        item_v = self.value_item(self.item_value) * self.item_value_valid[:, None]
        scale = float(np.sqrt(self.rho))
        user = torch.cat(
            [
                self.E_u.weight,
                scale * self.pi_n[:, None] * user_n,
                scale * self.pi_v[:, None] * user_v,
            ],
            dim=1,
        )
        item = torch.cat(
            [self.E_i.weight, scale * item_n, scale * item_v], dim=1
        )
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
        # The common evaluator expects external value slots.  N/V already live
        # inside the propagated embedding, so both compatibility slots are zero.
        user, item = self.propagate()
        return (
            user,
            item,
            user.new_zeros((self.n_users, 1)),
            item.new_zeros((self.n_items, 1)),
        )

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
        positive_score = (user[users] * item[positives]).sum(1)
        negative_score = (user[users] * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        return loss, {
            "bpr": float(bpr.detach()),
            "objective": "plain_bpr",
            "p_correct": float((positive_score > negative_score).float().mean().detach()),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict:
        user0, item0 = self.layer0_embeddings()
        n_start = self.id_dim
        v_start = n_start + self.axis_dim
        return {
            "rho": self.rho,
            "learned_global_axis_weight": False,
            "total_dim": self.total_dim,
            "pi_n_mean": float(self.pi_n.mean()),
            "pi_n_std": float(self.pi_n.std(unbiased=False)),
            "pi_v_mean": float(self.pi_v.mean()),
            "pi_v_std": float(self.pi_v.std(unbiased=False)),
            "composition_sum_mean": float((self.pi_n + self.pi_v).mean()),
            "mean_user_n_block_norm": float(user0[:, n_start:v_start].norm(dim=1).mean()),
            "mean_user_v_block_norm": float(user0[:, v_start:].norm(dim=1).mean()),
            "mean_item_n_block_norm": float(item0[:, n_start:v_start].norm(dim=1).mean()),
            "mean_item_v_block_norm": float(item0[:, v_start:].norm(dim=1).mean()),
            "mean_user_norm": float(user0.norm(dim=1).mean()),
            "mean_item_norm": float(item0.norm(dim=1).mean()),
        }
