"""CLV-conditioned category/within-category-price history for one LightGCN.

The auxiliary block is deliberately small and structured.  A user's train-only
distinct-item history is represented in a learned category basis and in the
same basis weighted by within-category price position.  The historical N/V
proxy does not create two independent scores: a ten-parameter softmax mixer
chooses the relative use of the two relations before they enter layer 0.

For training, the positive item is removed exactly from the user's layer-0
history.  LightGCN is linear, so for two propagation layers the resulting
pair-specific final embeddings can be obtained with a closed-form correction;
the full graph does not have to be recomputed for every positive edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConditionedHistoryFeatures:
    """Train-only fixed inputs used by the jointly learned representation."""

    user_state: np.ndarray
    q_n: np.ndarray
    q_v: np.ndarray
    n_valid: np.ndarray
    v_valid: np.ndarray
    auxiliary_valid: np.ndarray
    unique_item_count: np.ndarray
    history_users: np.ndarray
    history_categories: np.ndarray
    history_category_share: np.ndarray
    history_price_signal: np.ndarray
    item_category: np.ndarray
    item_price_percentile: np.ndarray
    item_price_signal: np.ndarray
    diagnostics: dict[str, float]


def _midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    result = np.zeros(values.shape, dtype=np.float32)
    if not valid.any():
        return result
    ranks = pd.Series(values[valid]).rank(method="average").to_numpy(np.float64)
    result[valid] = ((ranks - 0.5) / len(ranks)).astype(np.float32)
    return result


def _mean_available(columns: list[np.ndarray], masks: list[np.ndarray]):
    if len(columns) != len(masks) or not columns:
        raise ValueError("평균낼 값과 유효성 마스크가 일치해야 합니다")
    values = np.column_stack(columns).astype(np.float64)
    valid = np.column_stack(masks).astype(bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0.0).sum(axis=1)
    result = np.divide(total, count, out=np.zeros(len(values)), where=count > 0)
    return result.astype(np.float32), count > 0


def _time_as_days(values: pd.Series, is_date: bool) -> np.ndarray:
    if is_date:
        timestamp = pd.to_datetime(values, errors="raise")
        origin = timestamp.min()
        return (
            (timestamp - origin).dt.total_seconds().div(86400.0).to_numpy(np.float64)
        )
    return pd.to_numeric(values, errors="raise").to_numpy(np.float64)


def _mode_int(values: pd.Series) -> int:
    mode = values.mode(dropna=True)
    return int(mode.iat[0]) if len(mode) else int(values.iloc[0])


def build_conditioned_history_features(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    n_categories: int,
    is_date: bool,
) -> ConditionedHistoryFeatures:
    """Build behavior-composite N/V and structured history from train only."""

    required = {"u_idx", "i_idx", "cat_idx", "b_raw", "t", "v", "up"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"CLV 조건부 구매이력 입력 열 누락: {sorted(missing)}")
    if min(n_users, n_items, n_categories) <= 0:
        raise ValueError("사용자·상품·상품군 수는 양수여야 합니다")

    frame = train.loc[:, sorted(required)].copy()
    frame["u_idx"] = frame["u_idx"].astype(np.int64)
    frame["i_idx"] = frame["i_idx"].astype(np.int64)
    frame["cat_idx"] = frame["cat_idx"].astype(np.int64)
    if not frame["u_idx"].between(0, n_users - 1).all():
        raise ValueError("u_idx가 사용자 범위를 벗어났습니다")
    if not frame["i_idx"].between(0, n_items - 1).all():
        raise ValueError("i_idx가 상품 범위를 벗어났습니다")
    if not frame["cat_idx"].between(0, n_categories - 1).all():
        raise ValueError("cat_idx가 상품군 범위를 벗어났습니다")
    frame["positive_value"] = pd.to_numeric(frame["v"], errors="coerce").clip(
        lower=0.0
    )
    frame["unit_price"] = pd.to_numeric(frame["up"], errors="coerce")
    frame["time_days"] = _time_as_days(frame["t"], is_date)

    # One price and one category per catalog item, estimated from train only.
    item_table = (
        frame.groupby("i_idx", sort=True)
        .agg(
            category=("cat_idx", _mode_int),
            mean_price=("unit_price", "mean"),
        )
        .reset_index()
    )
    item_category = np.zeros(n_items, dtype=np.int64)
    item_category[item_table["i_idx"].to_numpy(np.int64)] = item_table[
        "category"
    ].to_numpy(np.int64)
    item_price_percentile = np.full(n_items, 0.5, dtype=np.float32)
    for _, group in item_table.groupby("category", sort=False):
        good = np.isfinite(group["mean_price"].to_numpy(np.float64))
        if not good.any():
            continue
        indices = group.loc[good, "i_idx"].to_numpy(np.int64)
        prices = group.loc[good, "mean_price"].to_numpy(np.float64)
        ranks = pd.Series(prices).rank(method="average").to_numpy(np.float64)
        item_price_percentile[indices] = ((ranks - 0.5) / len(ranks)).astype(
            np.float32
        )
    item_price_signal = (2.0 * item_price_percentile - 1.0).astype(np.float32)

    # Transactions are dataset-specific upstream: BASKET_ID for Dunnhumby and
    # customer-date bundles for H&M.  The same aggregation therefore works here.
    transaction = (
        frame.groupby(["u_idx", "b_raw"], sort=True)
        .agg(time_days=("time_days", "max"), amount=("positive_value", "sum"))
        .reset_index()
    )
    user_tx = transaction.groupby("u_idx", sort=False)
    basket_count_s = user_tx.size()
    first_s = user_tx["time_days"].min()
    last_s = user_tx["time_days"].max()
    amount_s = user_tx["amount"].sum()
    train_end = float(frame["time_days"].max())

    basket_count = np.zeros(n_users, dtype=np.float64)
    first = np.full(n_users, np.nan, dtype=np.float64)
    last = np.full(n_users, np.nan, dtype=np.float64)
    total_amount = np.zeros(n_users, dtype=np.float64)
    ids = basket_count_s.index.to_numpy(np.int64)
    basket_count[ids] = basket_count_s.to_numpy(np.float64)
    first[ids] = first_s.to_numpy(np.float64)
    last[ids] = last_s.to_numpy(np.float64)
    total_amount[ids] = amount_s.to_numpy(np.float64)
    observed_days = np.maximum(train_end - first + 1.0, 1.0)
    recency_days = np.maximum(train_end - last, 0.0)

    ordered = transaction.sort_values(["u_idx", "time_days", "b_raw"], kind="stable")
    ordered["gap"] = ordered.groupby("u_idx", sort=False)["time_days"].diff()
    mean_gap_s = ordered.groupby("u_idx", sort=False)["gap"].mean()
    mean_gap = np.full(n_users, np.nan, dtype=np.float64)
    mean_gap[mean_gap_s.index.to_numpy(np.int64)] = mean_gap_s.to_numpy(np.float64)

    activity_base_valid = (
        (basket_count > 0) & np.isfinite(observed_days) & np.isfinite(recency_days)
    )
    frequency_raw = np.divide(
        np.maximum(basket_count - 1.0, 0.0),
        observed_days,
        out=np.zeros(n_users, dtype=np.float64),
        where=activity_base_valid,
    )
    continuation_raw = np.zeros(n_users, dtype=np.float64)
    continuation_raw[activity_base_valid] = 1.0 - np.minimum(
        recency_days[activity_base_valid] / observed_days[activity_base_valid], 1.0
    )
    gap_valid = np.isfinite(mean_gap) & (mean_gap >= 0.0)
    inverse_gap_raw = np.zeros(n_users, dtype=np.float64)
    inverse_gap_raw[gap_valid] = 1.0 / (1.0 + mean_gap[gap_valid])

    frequency = _midrank_percentile(frequency_raw, activity_base_valid)
    continuation = _midrank_percentile(continuation_raw, activity_base_valid)
    inverse_gap = _midrank_percentile(inverse_gap_raw, gap_valid)
    q_n, n_valid = _mean_available(
        [frequency, continuation, inverse_gap],
        [activity_base_valid, activity_base_valid, gap_valid],
    )

    aov = np.divide(
        total_amount,
        basket_count,
        out=np.full(n_users, np.nan, dtype=np.float64),
        where=basket_count > 0,
    )
    aov_valid = np.isfinite(aov)
    aov_pct = _midrank_percentile(aov, aov_valid)

    distinct_lines = frame.drop_duplicates(["u_idx", "b_raw", "i_idx"]).copy()
    distinct_lines["item_price_percentile"] = item_price_percentile[
        distinct_lines["i_idx"].to_numpy(np.int64)
    ]
    tx_price = distinct_lines.groupby(["u_idx", "b_raw"], sort=False)[
        "item_price_percentile"
    ].mean()
    user_price_s = tx_price.groupby(level="u_idx", sort=False).mean()
    user_price = np.full(n_users, np.nan, dtype=np.float64)
    user_price[user_price_s.index.to_numpy(np.int64)] = user_price_s.to_numpy(
        np.float64
    )
    price_tendency_valid = np.isfinite(user_price)
    price_tendency_pct = _midrank_percentile(user_price, price_tendency_valid)
    q_v, v_valid = _mean_available(
        [aov_pct, price_tendency_pct], [aov_valid, price_tendency_valid]
    )

    user_state = np.column_stack(
        [q_n, q_v, q_n * q_v, q_n - q_v]
    ).astype(np.float32)

    # Distinct-item category distribution and category-specific price tendency.
    unique_pairs = frame.drop_duplicates(["u_idx", "i_idx"]).loc[
        :, ["u_idx", "i_idx"]
    ]
    unique_pairs["category"] = item_category[
        unique_pairs["i_idx"].to_numpy(np.int64)
    ]
    unique_pairs["price_signal"] = item_price_signal[
        unique_pairs["i_idx"].to_numpy(np.int64)
    ]
    unique_item_count = np.bincount(
        unique_pairs["u_idx"].to_numpy(np.int64), minlength=n_users
    ).astype(np.int64)
    by_category = (
        unique_pairs.groupby(["u_idx", "category"], sort=True)
        .agg(item_count=("i_idx", "size"), price_sum=("price_signal", "sum"))
        .reset_index()
    )
    history_users = by_category["u_idx"].to_numpy(np.int64)
    history_categories = by_category["category"].to_numpy(np.int64)
    denominator = unique_item_count[history_users].astype(np.float64)
    history_category_share = np.divide(
        by_category["item_count"].to_numpy(np.float64),
        denominator,
        out=np.zeros(len(by_category), dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)
    history_price_signal = np.divide(
        by_category["price_sum"].to_numpy(np.float64),
        denominator,
        out=np.zeros(len(by_category), dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)
    row_sum = np.bincount(
        history_users, weights=history_category_share, minlength=n_users
    )
    history_valid = unique_item_count > 0
    auxiliary_valid = n_valid & v_valid & (unique_item_count >= 2)
    diagnostics = {
        "train_end_time_days": train_end,
        "n_valid_user_share": float(n_valid.mean()),
        "v_valid_user_share": float(v_valid.mean()),
        "both_axis_valid_user_share": float((n_valid & v_valid).mean()),
        "single_unique_item_user_count": int((unique_item_count == 1).sum()),
        "single_unique_item_user_share": float((unique_item_count == 1).mean()),
        "auxiliary_valid_user_share": float(auxiliary_valid.mean()),
        "category_history_row_sum_max_error": float(
            np.abs(row_sum[history_valid] - 1.0).max() if history_valid.any() else 0.0
        ),
        "item_price_percentile_min": float(item_price_percentile.min()),
        "item_price_percentile_max": float(item_price_percentile.max()),
        "item_price_signal_min": float(item_price_signal.min()),
        "item_price_signal_max": float(item_price_signal.max()),
        "q_n_mean": float(q_n[n_valid].mean()) if n_valid.any() else 0.0,
        "q_n_std": float(q_n[n_valid].std()) if n_valid.any() else 0.0,
        "q_v_mean": float(q_v[v_valid].mean()) if v_valid.any() else 0.0,
        "q_v_std": float(q_v[v_valid].std()) if v_valid.any() else 0.0,
    }
    return ConditionedHistoryFeatures(
        user_state=user_state,
        q_n=q_n,
        q_v=q_v,
        n_valid=n_valid,
        v_valid=v_valid,
        auxiliary_valid=auxiliary_valid,
        unique_item_count=unique_item_count,
        history_users=history_users,
        history_categories=history_categories,
        history_category_share=history_category_share,
        history_price_signal=history_price_signal,
        item_category=item_category,
        item_price_percentile=item_price_percentile,
        item_price_signal=item_price_signal,
        diagnostics=diagnostics,
    )


class ConditionedCategoryPriceHistoryLightGCN(nn.Module):
    """One 72-dimensional LightGCN with a CLV-conditioned auxiliary block."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_categories: int,
        features: ConditionedHistoryFeatures,
        edge_users: np.ndarray,
        edge_items: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        category_dim: int = 4,
        n_layers: int = 2,
        rho: float = 0.1,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, n_categories, id_dim, category_dim) <= 0:
            raise ValueError("사용자·상품·상품군·임베딩 차원은 양수여야 합니다")
        if n_layers != 2:
            raise ValueError("정확한 leave-one-out 보정은 고정된 2층 전파만 지원합니다")
        if not 0.0 <= rho <= 0.1:
            raise ValueError("rho는 0 이상 0.1 이하여야 합니다")
        if features.user_state.shape != (n_users, 4):
            raise ValueError("사용자 CLV 상태는 [n_users,4]여야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_categories = int(n_categories)
        self.id_dim = int(id_dim)
        self.category_dim = int(category_dim)
        self.n_layers = int(n_layers)
        self.rho = float(rho)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        self.category_embedding = nn.Embedding(n_categories, category_dim)
        self.condition_mixer = nn.Linear(4, 2)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        nn.init.normal_(self.category_embedding.weight, std=0.1)
        nn.init.zeros_(self.condition_mixer.weight)
        nn.init.zeros_(self.condition_mixer.bias)

        history_indices = torch.from_numpy(
            np.stack([features.history_users, features.history_categories])
        )
        history_shape = (n_users, n_categories)
        category_history = torch.sparse_coo_tensor(
            history_indices,
            torch.from_numpy(features.history_category_share),
            history_shape,
        ).coalesce()
        price_history = torch.sparse_coo_tensor(
            history_indices,
            torch.from_numpy(features.history_price_signal),
            history_shape,
        ).coalesce()
        self.register_buffer("category_history", category_history, persistent=False)
        self.register_buffer("price_history", price_history, persistent=False)
        self.register_buffer(
            "user_state", torch.from_numpy(features.user_state), persistent=True
        )
        self.register_buffer(
            "auxiliary_valid",
            torch.from_numpy(features.auxiliary_valid.astype(np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "unique_item_count",
            torch.from_numpy(features.unique_item_count.astype(np.float32)),
            persistent=True,
        )
        self.register_buffer(
            "item_category",
            torch.from_numpy(features.item_category.astype(np.int64)),
            persistent=True,
        )
        self.register_buffer(
            "item_price_signal",
            torch.from_numpy(features.item_price_signal.astype(np.float32)),
            persistent=True,
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

        edge_users = np.asarray(edge_users, dtype=np.int64)
        edge_items = np.asarray(edge_items, dtype=np.int64)
        if edge_users.shape != edge_items.shape or not len(edge_users):
            raise ValueError("학습 이진엣지 사용자·상품 배열이 필요합니다")
        user_degree = np.bincount(edge_users, minlength=n_users).astype(np.float64)
        item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
        normalized = 1.0 / np.sqrt(
            user_degree[edge_users] * item_degree[edge_items]
        )
        edge_keys = edge_users * np.int64(n_items) + edge_items
        order = np.argsort(edge_keys, kind="stable")
        self.register_buffer(
            "edge_keys", torch.from_numpy(edge_keys[order]), persistent=False
        )
        self.register_buffer(
            "edge_normalized_weight",
            torch.from_numpy(normalized[order].astype(np.float32)),
            persistent=False,
        )
        squared_mass = np.bincount(
            edge_users, weights=normalized**2, minlength=n_users
        )
        user_self = ((1.0 + squared_mass) / 3.0).astype(np.float32)
        self.register_buffer(
            "user_loo_propagation_coefficient",
            torch.from_numpy(user_self),
            persistent=False,
        )
        self.feature_diagnostics = dict(features.diagnostics)

    @property
    def total_dim(self) -> int:
        return self.id_dim + 2 * self.category_dim

    @staticmethod
    def _unit_rows(values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values, dim=1, eps=1e-8)

    def _gate(self) -> torch.Tensor:
        return torch.softmax(self.condition_mixer(self.user_state), dim=1)

    def _layer0_blocks(self):
        category = self._unit_rows(self.category_embedding.weight)
        history_category = torch.sparse.mm(self.category_history, category)
        history_price = torch.sparse.mm(self.price_history, category)
        valid = self.auxiliary_valid[:, None]
        history_category = history_category * valid
        history_price = history_price * valid
        gate = self._gate()
        user_aux = torch.cat(
            [
                gate[:, :1] * history_category,
                gate[:, 1:] * history_price,
            ],
            dim=1,
        )
        item_category = category[self.item_category]
        item_price = self.item_price_signal[:, None] * item_category
        item_aux = torch.cat([item_category, item_price], dim=1)
        return user_aux, item_aux, history_category, history_price, category, gate

    def _propagated_blocks(self):
        user_aux, item_aux, hcat, hprice, category, gate = self._layer0_blocks()
        scale = float(np.sqrt(self.rho))
        user0 = torch.cat([self.E_u.weight, scale * user_aux], dim=1)
        item0 = torch.cat([self.E_i.weight, scale * item_aux], dim=1)
        current = torch.cat([user0, item0], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return (
            total[: self.n_users],
            total[self.n_users :],
            hcat,
            hprice,
            category,
            gate,
        )

    def embeddings(self, need_value: bool = True):
        user, item, *_ = self._propagated_blocks()
        return (
            user,
            item,
            user.new_zeros((self.n_users, 1)),
            item.new_zeros((self.n_items, 1)),
        )

    def id_only_embeddings(self):
        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def _positive_edge_weight(
        self, users: torch.Tensor, positives: torch.Tensor
    ) -> torch.Tensor:
        keys = users * self.n_items + positives
        positions = torch.searchsorted(self.edge_keys, keys)
        clipped = positions.clamp(max=len(self.edge_keys) - 1)
        if not torch.equal(self.edge_keys[clipped], keys):
            raise RuntimeError("학습 positive가 이진 학습그래프에 없습니다")
        return self.edge_normalized_weight[clipped]

    def _leave_one_out_auxiliary(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        hcat: torch.Tensor,
        hprice: torch.Tensor,
        category: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        count = self.unique_item_count[users]
        positive_category = self.item_category[positives]
        positive_category_vector = category[positive_category]
        positive_price_vector = (
            self.item_price_signal[positives, None] * positive_category_vector
        )
        remaining = (count - 1.0).clamp_min(1.0)
        loo_category = (
            count[:, None] * hcat[users] - positive_category_vector
        ) / remaining[:, None]
        loo_price = (
            count[:, None] * hprice[users] - positive_price_vector
        ) / remaining[:, None]
        eligible = (count > 1.0).float()[:, None] * self.auxiliary_valid[users, None]
        loo_category = loo_category * eligible
        loo_price = loo_price * eligible
        return torch.cat(
            [
                gate[users, :1] * loo_category,
                gate[users, 1:] * loo_price,
            ],
            dim=1,
        )

    def _pair_embeddings_with_exact_loo(
        self, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ):
        full_user, full_item, hcat, hprice, category, gate = self._propagated_blocks()
        full_aux = torch.cat(
            [
                gate[users, :1] * hcat[users],
                gate[users, 1:] * hprice[users],
            ],
            dim=1,
        )
        loo_aux = self._leave_one_out_auxiliary(
            users, positives, hcat, hprice, category, gate
        )
        delta_aux = float(np.sqrt(self.rho)) * (loo_aux - full_aux)
        user_pair = full_user[users].clone()
        positive_pair = full_item[positives].clone()
        negative_pair = full_item[negatives]
        user_pair[:, self.id_dim :] = (
            user_pair[:, self.id_dim :]
            + self.user_loo_propagation_coefficient[users, None] * delta_aux
        )
        positive_pair[:, self.id_dim :] = (
            positive_pair[:, self.id_dim :]
            + (self._positive_edge_weight(users, positives) / 3.0)[:, None]
            * delta_aux
        )
        return user_pair, positive_pair, negative_pair

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        category_rows = torch.cat(
            [self.item_category[positives], self.item_category[negatives]]
        ).unique()
        tables = (
            self.E_u.weight[users],
            self.E_i.weight[positives],
            self.E_i.weight[negatives],
            self.category_embedding.weight[category_rows],
        )
        return self.pref_reg * sum(table.pow(2).sum() for table in tables) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        if lam:
            raise ValueError("M2 표현 실험에 M4 lambda를 넣을 수 없습니다")
        user, positive, negative = self._pair_embeddings_with_exact_loo(
            users, positives, negatives
        )
        positive_score = (user * positive).sum(1)
        negative_score = (user * negative).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        return loss, {
            "bpr": float(bpr.detach()),
            "p_correct": float((positive_score > negative_score).float().mean().detach()),
            "objective": "plain_bpr",
        }

    @staticmethod
    def _corr(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.float() - left.float().mean()
        right = right.float() - right.float().mean()
        denom = left.norm() * right.norm()
        return float((left * right).sum() / denom) if denom > 0 else 0.0

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float]:
        user, item, hcat, hprice, _, gate = self._propagated_blocks()
        valid = self.auxiliary_valid.bool()
        alpha = gate[valid, 0]
        beta = gate[valid, 1]
        user_id, item_id = self.id_only_embeddings()
        user_aux = user[:, self.id_dim :]
        item_aux = item[:, self.id_dim :]
        sample_users = torch.where(valid)[0][: min(512, int(valid.sum()))]
        sample_items = torch.arange(
            min(2048, self.n_items), device=item.device, dtype=torch.long
        )
        id_scores = user_id[sample_users] @ item_id[sample_items].T
        aux_scores = user_aux[sample_users] @ item_aux[sample_items].T
        id_std = id_scores.std()
        aux_std = aux_scores.std()
        normalized_aux = F.normalize(user_aux[valid], dim=1, eps=1e-8)
        n_valid_aux = len(normalized_aux)
        if n_valid_aux > 1:
            cosine_sum = normalized_aux.sum(0).pow(2).sum() - n_valid_aux
            mean_pair_cosine = cosine_sum / (n_valid_aux * (n_valid_aux - 1))
        else:
            mean_pair_cosine = user_aux.new_zeros(())
        state = self.user_state[valid]
        result = {
            "rho": self.rho,
            "total_dim": self.total_dim,
            "alpha_mean": float(alpha.mean()) if len(alpha) else 0.0,
            "alpha_std": float(alpha.std()) if len(alpha) else 0.0,
            "alpha_min": float(alpha.min()) if len(alpha) else 0.0,
            "alpha_max": float(alpha.max()) if len(alpha) else 0.0,
            "beta_mean": float(beta.mean()) if len(beta) else 0.0,
            "beta_std": float(beta.std()) if len(beta) else 0.0,
            "auxiliary_score_std_ratio_to_id": float(aux_std / id_std.clamp_min(1e-8)),
            "category_history_mean_norm": float(hcat[valid].norm(dim=1).mean())
            if valid.any()
            else 0.0,
            "price_history_mean_norm": float(hprice[valid].norm(dim=1).mean())
            if valid.any()
            else 0.0,
            "user_auxiliary_mean_pair_cosine": float(mean_pair_cosine),
            "condition_mixer_weight_norm": float(self.condition_mixer.weight.norm()),
            "condition_mixer_bias_norm": float(self.condition_mixer.bias.norm()),
        }
        names = ("n_hat", "v_hat", "clv_proxy", "n_minus_v")
        for column, name in enumerate(names):
            result[f"alpha_{name}_correlation"] = self._corr(alpha, state[:, column])
        return self.feature_diagnostics | result

    def epoch_training_diagnostics(self) -> dict[str, float]:
        gate = self._gate()
        valid = self.auxiliary_valid.bool()
        alpha = gate[valid, 0]
        return {
            "alpha_mean": float(alpha.mean().detach()) if len(alpha) else 0.0,
            "alpha_std": float(alpha.std().detach()) if len(alpha) else 0.0,
            "condition_mixer_weight_norm": float(
                self.condition_mixer.weight.norm().detach()
            ),
        }

    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            gradient = parameter.grad
            return 0.0 if gradient is None else float(gradient.norm().detach())

        return {
            "user_id_gradient_norm": norm(self.E_u.weight),
            "item_id_gradient_norm": norm(self.E_i.weight),
            "category_embedding_gradient_norm": norm(self.category_embedding.weight),
            "condition_mixer_gradient_norm": norm(self.condition_mixer.weight),
        }
