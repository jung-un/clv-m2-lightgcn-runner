"""Train-only feature artifacts for CLV-conditioned embedding experts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_residual as residual


@dataclass(frozen=True)
class UserProfileArtifact:
    values: np.ndarray
    valid_user: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ItemProfileArtifact:
    numeric: np.ndarray
    category_ids: np.ndarray
    valid_item: np.ndarray
    numeric_names: tuple[str, ...]
    n_categories: int


def compose_user_profiles(
    artifact: residual.EncoderArtifact,
    snapshot: residual.AnchorExamples,
    device: torch.device,
) -> UserProfileArtifact:
    """Combine behavior, validity, and future-value representations per user."""
    transformed = residual.transform_features(snapshot, artifact.transform)
    x = torch.as_tensor(transformed, dtype=torch.float32, device=device)
    artifact.model.eval()
    with torch.no_grad():
        h, purchase_logit, log_amount = artifact.model(x)
        purchase_probability = torch.sigmoid(purchase_logit)
        log_ev = torch.log1p(
            purchase_probability * torch.expm1(log_amount).clamp_min(0.0)
        )
    local = np.concatenate(
        [
            transformed,
            h.detach().cpu().numpy(),
            purchase_probability.detach().cpu().numpy()[:, None],
            log_amount.detach().cpu().numpy()[:, None],
            log_ev.detach().cpu().numpy()[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    n_users = int(artifact.h_all.shape[0])
    user_ids = np.asarray(snapshot.user_ids, dtype=np.int64)
    if len(user_ids) and (user_ids.min() < 0 or user_ids.max() >= n_users):
        raise ValueError("snapshot user_ids가 encoder n_users 범위를 벗어났습니다")
    values = np.zeros((n_users, local.shape[1]), dtype=np.float32)
    valid_user = np.zeros(n_users, dtype=bool)
    values[user_ids] = local
    valid_user[user_ids] = True
    feature_names = residual.NUMERIC_FEATURES + tuple(
        f"valid_{name}" for name in residual.NUMERIC_FEATURES
    )
    feature_names += tuple(f"encoder_h_{idx}" for idx in range(16))
    feature_names += (
        "future_purchase_probability",
        "future_log_amount",
        "future_log_ev",
    )
    if values.shape[1] != len(feature_names):
        raise RuntimeError("user profile feature schema와 행렬 차원이 일치하지 않습니다")
    if not np.isfinite(values).all():
        raise ValueError("user profile에 유한하지 않은 값이 있습니다")
    return UserProfileArtifact(values, valid_user, feature_names)


def _modal_category(values: pd.Series):
    mode = values.mode(dropna=True)
    return mode.iat[0] if len(mode) else -1


def build_item_profiles(train: pd.DataFrame, n_items: int) -> ItemProfileArtifact:
    """Build economic and behavior features from the supplied training rows only."""
    required = {"u_idx", "i_idx", "cat_idx", "up"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"item profile 입력 열 누락: {sorted(missing)}")
    if n_items <= 0:
        raise ValueError("n_items는 양수여야 합니다")
    item_ids = train["i_idx"].to_numpy(dtype=np.int64)
    if len(item_ids) and (item_ids.min() < 0 or item_ids.max() >= n_items):
        raise ValueError("i_idx가 n_items 범위를 벗어났습니다")

    item = train.groupby("i_idx", sort=True).agg(
        price=("up", "mean"),
        rows=("i_idx", "size"),
        users=("u_idx", "nunique"),
        category=("cat_idx", _modal_category),
    )
    pair_counts = train.groupby(["u_idx", "i_idx"], sort=False).size()
    repeat_share = pair_counts.gt(1).groupby(level="i_idx").mean()
    item["repeat_share"] = repeat_share.reindex(item.index, fill_value=0.0)
    item["price_percentile"] = item["price"].rank(pct=True, method="average")
    item["category_price_percentile"] = item.groupby("category")["price"].rank(
        pct=True, method="average"
    )
    item["user_percentile"] = item["users"].rank(pct=True, method="average")
    raw = np.column_stack(
        [
            item["price_percentile"],
            item["category_price_percentile"],
            np.log1p(item["rows"].to_numpy(dtype=np.float64)),
            item["user_percentile"],
            item["repeat_share"],
            np.log1p(np.maximum(item["price"].to_numpy(dtype=np.float64), 0.0)),
        ]
    ).astype(np.float32)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    raw = ((raw - mean) / std).astype(np.float32)

    numeric = np.zeros((n_items, raw.shape[1]), dtype=np.float32)
    valid_item = np.zeros(n_items, dtype=bool)
    observed_ids = item.index.to_numpy(dtype=np.int64)
    numeric[observed_ids] = raw
    valid_item[observed_ids] = True

    categories = pd.Index(sorted(pd.unique(item["category"])))
    category_ids = np.zeros(n_items, dtype=np.int64)
    category_ids[observed_ids] = categories.get_indexer(item["category"]) + 1
    numeric_names = (
        "price_percentile",
        "category_price_percentile",
        "log_train_rows",
        "unique_user_percentile",
        "repeat_purchase_share",
        "log_mean_price",
    )
    if not np.isfinite(numeric).all():
        raise ValueError("item profile에 유한하지 않은 값이 있습니다")
    return ItemProfileArtifact(
        numeric=numeric,
        category_ids=category_ids,
        valid_item=valid_item,
        numeric_names=numeric_names,
        n_categories=len(categories) + 1,
    )
