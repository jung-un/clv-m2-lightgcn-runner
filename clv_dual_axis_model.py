"""Fixed-gate dual-axis CLV embedding adapters for a frozen LightGCN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

from clv_moe_features import UserProfileArtifact


CONTROLS = frozenset(
    {"dual_clv_fixed", "dual_shuffled_user", "dual_adapter_only"}
)
GATE_SHAPES = ("high", "equal", "low")


@dataclass(frozen=True)
class DualItemProfile:
    activity: np.ndarray
    value: np.ndarray
    valid_item: np.ndarray
    activity_names: tuple[str, ...]
    value_names: tuple[str, ...]


def _midrank_percentile(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return np.zeros(0, np.float32)
    return ((rankdata(values, method="average") - 0.5) / len(values)).astype(
        np.float32
    )


def fixed_percentile_ranks(
    n_hat: np.ndarray, v_hat: np.ndarray, valid_user: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return midrank CLV-axis percentiles for valid train-history users."""
    n_hat = np.asarray(n_hat, dtype=np.float64)
    v_hat = np.asarray(v_hat, dtype=np.float64)
    valid = np.asarray(valid_user, dtype=bool)
    if n_hat.shape != v_hat.shape or n_hat.shape != valid.shape:
        raise ValueError("N_hat, V_hat, valid_user shape이 다릅니다")
    if not np.isfinite(n_hat[valid]).all() or not np.isfinite(v_hat[valid]).all():
        raise ValueError("유효 사용자의 N_hat/V_hat이 유한해야 합니다")
    q_n = np.zeros(len(valid), np.float32)
    q_v = np.zeros(len(valid), np.float32)
    q_n[valid] = _midrank_percentile(n_hat[valid])
    q_v[valid] = _midrank_percentile(v_hat[valid])
    return q_n, q_v


def apply_gate_shape(percentiles: np.ndarray, shape: str) -> np.ndarray:
    """Map percentiles to one of three pre-specified mean-one gate directions."""
    q = np.asarray(percentiles, dtype=np.float32)
    if shape == "high":
        return 2.0 * q
    if shape == "equal":
        return np.ones_like(q)
    if shape == "low":
        return 2.0 * (1.0 - q)
    raise ValueError(f"지원하지 않는 gate shape: {shape}")


def _modal(values: pd.Series):
    mode = values.mode(dropna=True)
    return mode.iat[0] if len(mode) else -1


def _standardize(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((raw - mean) / std).astype(np.float32)


def build_dual_item_profiles(
    train: pd.DataFrame, n_items: int, is_date: bool
) -> DualItemProfile:
    required = {"u_idx", "i_idx", "cat_idx", "t", "up", "v"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"dual item profile 입력 열 누락: {sorted(missing)}")
    if n_items <= 0:
        raise ValueError("n_items는 양수여야 합니다")
    item_ids = train.i_idx.to_numpy(dtype=np.int64)
    if len(item_ids) and (item_ids.min() < 0 or item_ids.max() >= n_items):
        raise ValueError("i_idx가 n_items 범위를 벗어났습니다")

    item = train.groupby("i_idx", sort=True).agg(
        rows=("i_idx", "size"),
        users=("u_idx", "nunique"),
        category=("cat_idx", _modal),
        mean_price=("up", "mean"),
    )
    pairs = train.groupby(["u_idx", "i_idx"], sort=False).size()
    item["repeat_share"] = (
        pairs.gt(1).groupby(level="i_idx").mean().reindex(item.index, fill_value=0.0)
    )

    dated = (
        train[["u_idx", "i_idx", "t"]]
        .drop_duplicates()
        .sort_values(["u_idx", "i_idx", "t"])
    )
    gap = dated.groupby(["u_idx", "i_idx"], sort=False)["t"].diff()
    if is_date:
        gap = gap.dt.total_seconds() / 86400.0
    dated = dated.assign(_gap=np.asarray(gap, dtype=float))
    repeat_gap = dated.groupby("i_idx", sort=False)["_gap"].median()
    item["repeat_gap"] = repeat_gap.reindex(item.index)
    item["repeat_gap_valid"] = item["repeat_gap"].notna().astype(np.float32)
    item["repeat_gap"] = item["repeat_gap"].fillna(0.0).clip(lower=0.0)

    item["price_percentile"] = _midrank_percentile(
        item.mean_price.to_numpy(dtype=float)
    )
    item["category_price_percentile"] = item.groupby("category")[
        "mean_price"
    ].rank(pct=True, method="average")

    transaction_keys = ["b_raw"] if "b_raw" in train.columns else ["u_idx", "t"]
    item_in_transaction = train.groupby(
        [*transaction_keys, "i_idx"], sort=False
    ).v.sum()
    transaction_levels = list(range(len(transaction_keys)))
    transaction_total = item_in_transaction.groupby(
        level=transaction_levels, sort=False
    ).transform("sum")
    transaction_share = item_in_transaction.div(
        transaction_total.where(transaction_total > 0)
    ).fillna(0.0)
    item["transaction_value_share"] = (
        transaction_share.groupby(level="i_idx").mean().reindex(item.index, fill_value=0)
    )

    activity_numeric = _standardize(
        np.column_stack(
            [
                item.repeat_share,
                np.log1p(item.repeat_gap),
            ]
        )
    )
    activity_raw = np.column_stack(
        [activity_numeric, item.repeat_gap_valid.to_numpy(np.float32)]
    ).astype(np.float32)
    value_raw = _standardize(
        np.column_stack(
            [
                item.price_percentile,
                item.category_price_percentile,
                np.log1p(np.maximum(item.mean_price.to_numpy(float), 0.0)),
                item.transaction_value_share,
            ]
        )
    )
    observed = item.index.to_numpy(dtype=np.int64)
    activity = np.zeros((n_items, activity_raw.shape[1]), np.float32)
    value = np.zeros((n_items, value_raw.shape[1]), np.float32)
    valid_item = np.zeros(n_items, bool)
    activity[observed] = activity_raw
    value[observed] = value_raw
    valid_item[observed] = True
    if not np.isfinite(activity).all() or not np.isfinite(value).all():
        raise ValueError("dual item profile에 유한하지 않은 값이 있습니다")
    return DualItemProfile(
        activity,
        value,
        valid_item,
        (
            "repeat_purchase_share",
            "log_median_repeat_gap",
            "repeat_gap_valid",
        ),
        (
            "price_percentile",
            "category_price_percentile",
            "log_mean_unit_price",
            "mean_transaction_value_share",
        ),
    )


class _AxisExpert(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, hidden: int, output: int):
        super().__init__()
        self.user = nn.Sequential(
            nn.Linear(user_dim, hidden), nn.GELU(), nn.Linear(hidden, output)
        )
        self.item = nn.Sequential(
            nn.Linear(item_dim, hidden), nn.GELU(), nn.Linear(hidden, output)
        )
        nn.init.normal_(self.user[-1].weight, std=0.01)
        nn.init.normal_(self.item[-1].weight, std=0.01)
        nn.init.zeros_(self.user[-1].bias)
        nn.init.zeros_(self.item[-1].bias)


def _profile_indices(names: tuple[str, ...], axis: str) -> list[int]:
    prefix = "repurchase_" if axis == "activity" else "monetary_"
    prediction = (
        "pred_log_future_transactions"
        if axis == "activity"
        else "pred_log_transaction_value"
    )
    indices = [
        index
        for index, name in enumerate(names)
        if name.startswith(prefix) or name == prediction
    ]
    if not indices:
        raise ValueError(f"user profile에 {axis} 축 feature가 없습니다")
    return indices


class CLVDualAxisEmbeddingModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        user_profile: UserProfileArtifact,
        item_profile: DualItemProfile,
        q_n: np.ndarray,
        q_v: np.ndarray,
        *,
        control: str = "dual_clv_fixed",
        seed: int = 42,
        hidden_dim: int = 32,
        expert_dim: int = 16,
    ):
        super().__init__()
        if control not in CONTROLS:
            raise ValueError(f"지원하지 않는 dual control: {control}")
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model = base_model
        self.control = control
        with torch.no_grad():
            base_user, base_item, *_ = base_model.embeddings(need_value=False)
        self.register_buffer("base_user", base_user.detach().clone(), persistent=False)
        self.register_buffer("base_item", base_item.detach().clone(), persistent=False)
        values = torch.as_tensor(user_profile.values, dtype=torch.float32)
        activity_idx = _profile_indices(user_profile.feature_names, "activity")
        value_idx = _profile_indices(user_profile.feature_names, "value")
        user_activity = values[:, activity_idx].clone()
        user_value = values[:, value_idx].clone()
        item_activity = torch.as_tensor(item_profile.activity, dtype=torch.float32)
        item_value = torch.as_tensor(item_profile.value, dtype=torch.float32)
        rank_n = torch.as_tensor(q_n, dtype=torch.float32).clone()
        rank_v = torch.as_tensor(q_v, dtype=torch.float32).clone()
        valid_user = torch.as_tensor(user_profile.valid_user, dtype=torch.bool)
        if len(rank_n) != len(values) or len(rank_v) != len(values):
            raise ValueError("gate와 user profile 행 수가 다릅니다")
        if control == "dual_shuffled_user":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            indices = torch.where(valid_user)[0]
            if len(indices) > 1:
                source = indices[torch.randperm(len(indices), generator=generator)]
                user_activity[indices] = user_activity[source].clone()
                user_value[indices] = user_value[source].clone()
                rank_n[indices] = rank_n[source].clone()
                rank_v[indices] = rank_v[source].clone()
        elif control == "dual_adapter_only":
            user_activity.zero_()
            user_value.zero_()
            item_activity.zero_()
            item_value.zero_()
            rank_n[valid_user] = 0.5
            rank_v[valid_user] = 0.5
        self.register_buffer("user_activity", user_activity)
        self.register_buffer("user_value", user_value)
        self.register_buffer("item_activity", item_activity)
        self.register_buffer("item_value", item_value)
        self.register_buffer("has_profile", valid_user)
        self.register_buffer(
            "valid_item", torch.as_tensor(item_profile.valid_item, dtype=torch.bool)
        )
        self.register_buffer("q_n", rank_n)
        self.register_buffer("q_v", rank_v)
        self.eval_gate_shape = "equal"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.activity_expert = _AxisExpert(
                self.base_user.shape[1] + user_activity.shape[1],
                self.base_item.shape[1] + item_activity.shape[1],
                hidden_dim,
                expert_dim,
            )
            self.value_expert = _AxisExpert(
                self.base_user.shape[1] + user_value.shape[1],
                self.base_item.shape[1] + item_value.shape[1],
                hidden_dim,
                expert_dim,
            )

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [*self.activity_expert.parameters(), *self.value_expert.parameters()]

    def set_gate_shape(self, shape: str) -> None:
        if shape not in GATE_SHAPES:
            raise ValueError(f"지원하지 않는 gate shape: {shape}")
        self.eval_gate_shape = shape

    def gate_values(self, shape: str | None = None):
        shape = shape or self.eval_gate_shape
        if shape not in GATE_SHAPES:
            raise ValueError(f"지원하지 않는 gate shape: {shape}")
        if self.control == "dual_adapter_only" or shape == "equal":
            g_n = torch.ones_like(self.q_n)
            g_v = torch.ones_like(self.q_v)
        elif shape == "high":
            g_n, g_v = 2.0 * self.q_n, 2.0 * self.q_v
        else:
            g_n, g_v = 2.0 * (1.0 - self.q_n), 2.0 * (1.0 - self.q_v)
        return g_n, g_v

    def _axis_embeddings(self, users=None, items=None):
        if users is None:
            users = torch.arange(len(self.base_user), device=self.base_user.device)
        if items is None:
            items = torch.arange(len(self.base_item), device=self.base_item.device)
        u_n = F.normalize(
            self.activity_expert.user(
                torch.cat([self.base_user[users], self.user_activity[users]], dim=1)
            ),
            dim=1,
        )
        i_n = F.normalize(
            self.activity_expert.item(
                torch.cat([self.base_item[items], self.item_activity[items]], dim=1)
            ),
            dim=1,
        ) * self.valid_item[items, None]
        u_v = F.normalize(
            self.value_expert.user(
                torch.cat([self.base_user[users], self.user_value[users]], dim=1)
            ),
            dim=1,
        )
        i_v = F.normalize(
            self.value_expert.item(
                torch.cat([self.base_item[items], self.item_value[items]], dim=1)
            ),
            dim=1,
        ) * self.valid_item[items, None]
        return u_n, i_n, u_v, i_v

    def base_score_all(self, users: torch.Tensor) -> torch.Tensor:
        return self.base_user[users] @ self.base_item.T

    def score_all(
        self, users: torch.Tensor, lam: float, gate_shape: str | None = None
    ) -> torch.Tensor:
        base = self.base_score_all(users)
        if float(lam) == 0.0:
            return base
        u_n, i_n, u_v, i_v = self._axis_embeddings(users=users)
        g_n, g_v = self.gate_values(gate_shape)
        residual = g_n[users, None] * (u_n @ i_n.T)
        residual += g_v[users, None] * (u_v @ i_v.T)
        return base + float(lam) * self.has_profile[users, None] * residual

    def score_pairs(self, users, items, lam: float, gate_shape: str | None = None):
        base = (self.base_user[users] * self.base_item[items]).sum(dim=1)
        if float(lam) == 0.0:
            return base
        u_n, i_n, u_v, i_v = self._axis_embeddings(users, items)
        g_n, g_v = self.gate_values(gate_shape)
        residual = g_n[users] * (u_n * i_n).sum(dim=1)
        residual += g_v[users] * (u_v * i_v).sum(dim=1)
        return base + float(lam) * self.has_profile[users] * residual

    def embeddings(self, need_value: bool = True):
        if not need_value:
            return self.base_user, self.base_item, None, None
        u_n, i_n, u_v, i_v = self._axis_embeddings()
        mask = self.has_profile[:, None]
        g_n, g_v = self.gate_values()
        value_user = torch.cat(
            [g_n[:, None] * u_n * mask, g_v[:, None] * u_v * mask], dim=1
        )
        value_item = torch.cat([i_n, i_v], dim=1)
        return self.base_user, self.base_item, value_user, value_item

    def bpr_loss(self, users, positives, negatives, lam: float = 1.0):
        positive = self.score_pairs(users, positives, lam, gate_shape="equal")
        negative = self.score_pairs(users, negatives, lam, gate_shape="equal")
        return -F.logsigmoid(positive - negative).mean()

    @torch.no_grad()
    def axis_diagnostics(self, gate_shape: str, max_users=256, max_items=512):
        users = torch.where(self.has_profile)[0][:max_users]
        items = torch.where(self.valid_item)[0][:max_items]
        if not len(users) or not len(items):
            return {
                "effective_n_ratio": float("nan"),
                "effective_v_ratio": float("nan"),
                "effective_total_ratio": float("nan"),
                "expert_score_corr": float("nan"),
                "expert_top10_jaccard": float("nan"),
            }
        base = self.base_user[users] @ self.base_item[items].T
        u_n, i_n, u_v, i_v = self._axis_embeddings(users, items)
        g_n, g_v = self.gate_values(gate_shape)
        score_n = g_n[users, None] * (u_n @ i_n.T)
        score_v = g_v[users, None] * (u_v @ i_v.T)
        base_std = base.std(unbiased=False).clamp_min(1e-12)
        flat_n, flat_v = score_n.flatten(), score_v.flatten()
        if flat_n.std(unbiased=False) > 0 and flat_v.std(unbiased=False) > 0:
            corr = torch.corrcoef(torch.stack([flat_n, flat_v]))[0, 1]
        else:
            corr = torch.tensor(float("nan"), device=base.device)
        k = min(10, len(items))
        top_n = score_n.topk(k, dim=1).indices
        top_v = score_v.topk(k, dim=1).indices
        overlaps = []
        for left, right in zip(top_n.cpu().tolist(), top_v.cpu().tolist()):
            intersection = len(set(left).intersection(right))
            overlaps.append(intersection / (2 * k - intersection))
        return {
            "effective_n_ratio": float(score_n.std(unbiased=False) / base_std),
            "effective_v_ratio": float(score_v.std(unbiased=False) / base_std),
            "effective_total_ratio": float(
                (score_n + score_v).std(unbiased=False) / base_std
            ),
            "expert_score_corr": float(corr),
            "expert_top10_jaccard": float(np.mean(overlaps)),
        }
