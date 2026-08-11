"""Future-value-supervised CLV residual for the v3 LightGCN runner.

This module deliberately keeps the recommender objective equal to M1's plain
BPR.  CLV supervision is used only to learn a user representation before the
recommender adapter is trained; CLV-weighted ranking losses belong to M4.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


CODE_VERSION = "clv-residual-v1.0"
NUMERIC_FEATURES = (
    "basket_count",
    "purchase_days",
    "recency_days",
    "total_value",
    "avg_basket_value",
    "unit_price_mean",
    "unit_price_median",
    "unit_price_p80",
    "premium_share",
    "repeat_pair_share",
    "category_count",
    "category_entropy",
    "gap_mean",
    "gap_std",
    "spend_trend",
    "observed_days",
)


@dataclass
class AnchorExamples:
    offset_days: int
    observation_start: object
    observation_end: object
    target_start: object
    target_end: object
    user_ids: np.ndarray
    numeric: np.ndarray
    valid: np.ndarray
    purchase_target: np.ndarray
    amount_target: np.ndarray
    transaction_target: np.ndarray | None = None
    mean_transaction_value_target: np.ndarray | None = None


@dataclass
class AnchorDataset:
    anchors: list[AnchorExamples]
    train_end: object
    n_users: int
    feature_names: tuple[str, ...] = NUMERIC_FEATURES


@dataclass
class FeatureTransform:
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...] = NUMERIC_FEATURES


@dataclass
class ResidualConfig:
    dataset: str = "dunnhumby"
    seed_list: tuple[int, ...] = (42,)
    input_days: int = 365
    target_days: int = 90
    anchor_offsets: tuple[int, ...] = (270, 180, 90)
    encoder_epochs: int = 100
    encoder_patience: int = 10
    encoder_batch_size: int = 1024
    encoder_lr: float = 1e-3
    adapter_epochs: int = 100
    adapter_patience: int = 20
    adapter_lr: float = 5e-4
    lambda_eval: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    accuracy_tolerance: float = 0.01
    include_constant_control: bool = True
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None
    m1_checkpoint_dir: str | None = None


@dataclass
class EncoderArtifact:
    model: nn.Module
    transform: FeatureTransform
    best_epoch: int
    diagnostics: dict
    h_all: np.ndarray
    ev_all: np.ndarray


def _delta(days: int, is_date: bool):
    return pd.Timedelta(days=days) if is_date else days


def _days_between(a, b, is_date: bool) -> float:
    d = a - b
    return float(d.days if is_date else d)


def _feature_row(
    g: pd.DataFrame, obs_end, obs_start, is_date: bool, premium_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)
    valid = np.ones(len(NUMERIC_FEATURES), dtype=bool)
    basket_keys = ["b_raw"] if "b_raw" in g.columns else ["t"]
    basket = g.groupby(basket_keys, sort=False).agg(
        bval=("v", "sum"), btime=("t", "max")
    )
    days = np.sort(g["t"].unique())
    gaps = np.diff(days)
    if is_date:
        gaps = np.asarray([x / np.timedelta64(1, "D") for x in gaps], dtype=float)
    else:
        gaps = gaps.astype(float)
    pair_counts = g.groupby("i_idx").size()
    cat_counts = g.groupby("cat_idx").size().to_numpy(dtype=float)
    probs = cat_counts / cat_counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    entropy_denom = math.log(len(cat_counts)) if len(cat_counts) > 1 else 0.0

    prev_start = obs_end - _delta(180, is_date)
    recent_start = obs_end - _delta(90, is_date)
    previous = float(g.loc[(g.t > prev_start) & (g.t <= recent_start), "v"].sum())
    recent = float(g.loc[g.t > recent_start, "v"].sum())

    vals = {
        "basket_count": float(len(basket)),
        "purchase_days": float(len(days)),
        "recency_days": _days_between(obs_end, g.t.max(), is_date),
        "total_value": float(g.v.sum()),
        "avg_basket_value": float(basket.bval.mean()),
        "unit_price_mean": float(g.up.mean()),
        "unit_price_median": float(g.up.median()),
        "unit_price_p80": float(g.up.quantile(0.8)),
        "premium_share": float((g.up >= premium_threshold).mean()),
        "repeat_pair_share": float(pair_counts[pair_counts > 1].sum() / len(g)),
        "category_count": float(len(cat_counts)),
        "category_entropy": entropy / entropy_denom if entropy_denom > 0 else 0.0,
        "gap_mean": float(gaps.mean()) if len(gaps) else 0.0,
        "gap_std": float(gaps.std()) if len(gaps) > 1 else 0.0,
        "spend_trend": recent / previous if previous > 0 else 0.0,
        "observed_days": min(_days_between(obs_end, g.t.min(), is_date) + 1, 365.0),
    }
    for idx, name in enumerate(NUMERIC_FEATURES):
        values[idx] = vals[name]
    valid[NUMERIC_FEATURES.index("category_entropy")] = len(cat_counts) > 1
    valid[NUMERIC_FEATURES.index("gap_mean")] = len(gaps) > 0
    valid[NUMERIC_FEATURES.index("gap_std")] = len(gaps) > 1
    valid[NUMERIC_FEATURES.index("spend_trend")] = previous > 0
    return values, valid


def _feature_matrix(
    obs: pd.DataFrame,
    users: np.ndarray,
    obs_end,
    is_date: bool,
    premium_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``_feature_row`` for full-scale datasets."""
    user_index = pd.Index(users, name="u_idx")
    grouped = obs.groupby("u_idx", sort=False)
    frame = pd.DataFrame(index=user_index)
    frame["purchase_days"] = grouped["t"].nunique()
    frame["total_value"] = grouped["v"].sum()
    frame["unit_price_mean"] = grouped["up"].mean()
    frame["unit_price_median"] = grouped["up"].median()
    frame["unit_price_p80"] = grouped["up"].quantile(0.8)
    first, last = grouped["t"].min(), grouped["t"].max()
    if is_date:
        frame["recency_days"] = (obs_end - last).dt.days
        frame["observed_days"] = ((obs_end - first).dt.days + 1).clip(upper=365)
    else:
        frame["recency_days"] = obs_end - last
        frame["observed_days"] = (obs_end - first + 1).clip(upper=365)

    basket_key = "b_raw" if "b_raw" in obs.columns else "t"
    baskets = obs.groupby(["u_idx", basket_key], sort=False)["v"].sum()
    basket_stats = baskets.groupby(level=0).agg(["size", "mean"])
    frame["basket_count"] = basket_stats["size"]
    frame["avg_basket_value"] = basket_stats["mean"]
    frame["premium_share"] = (
        obs.assign(_premium=(obs.up >= premium_threshold).astype(np.float32))
        .groupby("u_idx", sort=False)["_premium"]
        .mean()
    )

    line_count = grouped.size()
    pair_count = obs.groupby(["u_idx", "i_idx"], sort=False).size()
    repeated_lines = pair_count[pair_count > 1].groupby(level=0).sum()
    frame["repeat_pair_share"] = (
        repeated_lines.reindex(user_index, fill_value=0) / line_count
    )

    category_count = obs.groupby(["u_idx", "cat_idx"], sort=False).size().astype(float)
    category_probability = category_count / category_count.groupby(level=0).transform(
        "sum"
    )
    entropy = (
        (-(category_probability * np.log(category_probability + 1e-12)))
        .groupby(level=0)
        .sum()
    )
    n_categories = category_count.groupby(level=0).size()
    frame["category_count"] = n_categories
    frame["category_entropy"] = (
        entropy / np.log(n_categories).replace(0, np.nan)
    ).fillna(0.0)

    purchase_dates = obs[["u_idx", "t"]].drop_duplicates().sort_values(["u_idx", "t"])
    gaps = purchase_dates.groupby("u_idx", sort=False)["t"].diff()
    if is_date:
        gaps = gaps.dt.total_seconds() / 86400.0
    gap_group = purchase_dates.assign(_gap=gaps).groupby("u_idx", sort=False)["_gap"]
    gap_stats = pd.DataFrame(
        {
            "mean": gap_group.mean(),
            "std": gap_group.std(ddof=0),
            "count": gap_group.count(),
        }
    )
    frame["gap_mean"] = gap_stats["mean"].fillna(0.0)
    frame["gap_std"] = gap_stats["std"].fillna(0.0)

    previous_start = obs_end - _delta(180, is_date)
    recent_start = obs_end - _delta(90, is_date)
    previous = (
        obs.loc[(obs.t > previous_start) & (obs.t <= recent_start)]
        .groupby("u_idx")["v"]
        .sum()
    )
    recent = obs.loc[obs.t > recent_start].groupby("u_idx")["v"].sum()
    previous = previous.reindex(user_index, fill_value=0.0)
    recent = recent.reindex(user_index, fill_value=0.0)
    frame["spend_trend"] = recent.div(previous.where(previous > 0)).fillna(0.0)

    frame = frame.reindex(user_index).fillna(0.0)
    numeric = frame.loc[:, NUMERIC_FEATURES].to_numpy(np.float32)
    valid = np.ones_like(numeric, dtype=bool)
    valid[:, NUMERIC_FEATURES.index("category_entropy")] = (
        n_categories.reindex(user_index, fill_value=0).to_numpy() > 1
    )
    gap_count = gap_stats["count"].reindex(user_index, fill_value=0).to_numpy()
    valid[:, NUMERIC_FEATURES.index("gap_mean")] = gap_count > 0
    valid[:, NUMERIC_FEATURES.index("gap_std")] = gap_count > 1
    valid[:, NUMERIC_FEATURES.index("spend_trend")] = previous.to_numpy() > 0
    return numeric, valid


def build_anchor_examples(
    train: pd.DataFrame,
    n_users: int,
    is_date: bool,
    input_days: int = 365,
    target_days: int = 90,
    anchor_offsets: Sequence[int] = (270, 180, 90),
) -> AnchorDataset:
    required = input_days + max(anchor_offsets)
    train_end, train_start = train.t.max(), train.t.min()
    if _days_between(train_end, train_start, is_date) < required:
        raise ValueError(
            f"공식 train에 최소 {required}일(기본 635일)의 연속 관찰기간이 필요합니다"
        )
    anchors: list[AnchorExamples] = []
    for offset in anchor_offsets:
        obs_end = train_end - _delta(offset, is_date)
        obs_start = obs_end - _delta(input_days, is_date)
        target_start = obs_end + _delta(1, is_date)
        target_end = obs_end + _delta(target_days, is_date)
        obs = train[(train.t > obs_start) & (train.t <= obs_end)]
        target = train[(train.t >= target_start) & (train.t <= target_end)]
        users = np.sort(obs.u_idx.unique()).astype(np.int64)
        if len(users) == 0:
            raise ValueError(f"anchor T-{offset}에 관찰 고객이 없습니다")
        premium_threshold = float(obs.up.quantile(0.8))
        numeric, valid = _feature_matrix(
            obs, users, obs_end, is_date, premium_threshold
        )
        future = target.groupby("u_idx").v.sum().clip(lower=0.0)
        amount = np.asarray([future.get(u, 0.0) for u in users], dtype=np.float32)
        target_basket_key = "b_raw" if "b_raw" in target.columns else "t"
        future_transactions = target.groupby("u_idx")[target_basket_key].nunique()
        transaction_count = np.asarray(
            [future_transactions.get(u, 0.0) for u in users], dtype=np.float32
        )
        mean_transaction_value = np.divide(
            amount,
            transaction_count,
            out=np.zeros_like(amount),
            where=transaction_count > 0,
        )
        anchors.append(
            AnchorExamples(
                offset,
                obs_start,
                obs_end,
                target_start,
                target_end,
                users,
                numeric,
                valid,
                (amount > 0).astype(np.float32),
                amount,
                transaction_count,
                mean_transaction_value,
            )
        )
    return AnchorDataset(anchors=anchors, train_end=train_end, n_users=n_users)


def fit_feature_transform(anchors: Sequence[AnchorExamples]) -> FeatureTransform:
    x = np.concatenate([a.numeric for a in anchors], axis=0).astype(np.float64)
    mask = np.concatenate([a.valid for a in anchors], axis=0)
    mean = np.zeros(x.shape[1], dtype=np.float32)
    std = np.ones(x.shape[1], dtype=np.float32)
    log_features = set(NUMERIC_FEATURES) - {
        "premium_share",
        "repeat_pair_share",
        "category_entropy",
        "spend_trend",
    }
    for j, name in enumerate(NUMERIC_FEATURES):
        col = np.log1p(np.maximum(x[:, j], 0)) if name in log_features else x[:, j]
        good = mask[:, j] & np.isfinite(col)
        if good.any():
            mean[j] = col[good].mean()
            s = col[good].std()
            std[j] = s if s > 1e-8 else 1.0
    return FeatureTransform(mean, std)


def transform_features(
    anchor: AnchorExamples, transform: FeatureTransform
) -> np.ndarray:
    if tuple(transform.feature_names) != NUMERIC_FEATURES:
        raise ValueError("feature 이름·순서가 현재 schema와 다릅니다")
    x = anchor.numeric.astype(np.float32).copy()
    log_features = set(NUMERIC_FEATURES) - {
        "premium_share",
        "repeat_pair_share",
        "category_entropy",
        "spend_trend",
    }
    for j, name in enumerate(NUMERIC_FEATURES):
        if name in log_features:
            x[:, j] = np.log1p(np.maximum(x[:, j], 0))
    x = (x - transform.mean) / transform.std
    x[~anchor.valid] = 0.0
    return np.concatenate([x, anchor.valid.astype(np.float32)], axis=1).astype(
        np.float32
    )


def build_final_snapshot(
    train: pd.DataFrame, n_users: int, is_date: bool, input_days: int = 365
) -> AnchorExamples:
    end = train.t.max()
    start = end - _delta(input_days, is_date)
    obs = train[(train.t > start) & (train.t <= end)]
    users = np.sort(obs.u_idx.unique()).astype(np.int64)
    threshold = float(obs.up.quantile(0.8))
    numeric, valid = _feature_matrix(obs, users, end, is_date, threshold)
    zeros = np.zeros(len(users), np.float32)
    return AnchorExamples(
        0,
        start,
        end,
        end + _delta(1, is_date),
        end,
        users,
        numeric,
        valid,
        zeros,
        zeros,
    )


class FutureValueEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU()
        )
        self.purchase_head = nn.Linear(16, 1)
        self.amount_head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (
            h,
            self.purchase_head(h).squeeze(-1),
            F.softplus(self.amount_head(h).squeeze(-1)),
        )


def future_value_loss(
    purchase_logit,
    log_amount,
    purchase_target,
    amount_target,
    pos_weight: float | torch.Tensor,
):
    pw = torch.as_tensor(
        pos_weight, dtype=purchase_logit.dtype, device=purchase_logit.device
    )
    purchase_loss = F.binary_cross_entropy_with_logits(
        purchase_logit, purchase_target, pos_weight=pw
    )
    buyers = purchase_target > 0
    amount_loss = (
        F.huber_loss(log_amount[buyers], torch.log1p(amount_target[buyers]))
        if buyers.any()
        else purchase_logit.sum() * 0.0
    )
    return purchase_loss + amount_loss


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stack_anchors(anchors: Sequence[AnchorExamples], transform: FeatureTransform):
    x = np.concatenate([transform_features(a, transform) for a in anchors])
    y = np.concatenate([a.purchase_target for a in anchors])
    amount = np.concatenate([a.amount_target for a in anchors])
    return x, y, amount


def _train_encoder_epochs(model, x, y, amount, epochs, cfg, seed, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.encoder_lr)
    rng = np.random.default_rng(seed)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    pos_weight = negatives / positives if positives > 0 and negatives > 0 else 1.0
    for _ in range(epochs):
        model.train()
        for s in range(0, len(x), cfg.encoder_batch_size):
            if s == 0:
                order = rng.permutation(len(x))
            idx = order[s : s + cfg.encoder_batch_size]
            xb = torch.as_tensor(x[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            ab = torch.as_tensor(amount[idx], dtype=torch.float32, device=device)
            _, logit, pred_amount = model(xb)
            loss = future_value_loss(logit, pred_amount, yb, ab, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _encoder_diagnostics(model, anchor, transform, device):
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        mean_absolute_error,
        roc_auc_score,
    )

    x = torch.as_tensor(
        transform_features(anchor, transform), dtype=torch.float32, device=device
    )
    model.eval()
    with torch.no_grad():
        _, logit, log_amount = model(x)
        p = torch.sigmoid(logit).cpu().numpy()
        la = log_amount.cpu().numpy()
    y, amount = anchor.purchase_target, anchor.amount_target
    buyers = y > 0
    diag = {
        "roc_auc": float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_amount_mae": (
            float(mean_absolute_error(np.log1p(amount[buyers]), la[buyers]))
            if buyers.any()
            else float("nan")
        ),
    }
    ev = p * np.expm1(la)
    corr = spearmanr(ev, amount, nan_policy="omit").statistic if len(ev) > 1 else np.nan
    diag["ev_amount_spearman"] = float(corr) if np.isfinite(corr) else float("nan")
    bins = pd.qcut(
        pd.Series(ev).rank(method="first"),
        q=min(5, len(ev)),
        labels=False,
        duplicates="drop",
    )
    diag["value_bins"] = [
        {
            "bin": int(b),
            "n": int((bins == b).sum()),
            "actual_amount_mean": float(amount[bins == b].mean()),
            "purchase_rate": float(y[bins == b].mean()),
        }
        for b in sorted(pd.unique(bins))
    ]
    return diag


def train_future_value_encoder(
    dataset: AnchorDataset,
    final_snapshot: AnchorExamples,
    cfg: ResidualConfig,
    seed: int,
    device: torch.device | None = None,
) -> EncoderArtifact:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(dataset.anchors) != 3:
        raise ValueError(
            "encoder 설계는 학습 anchor 2개와 내부 validation anchor 1개를 요구합니다"
        )
    transform = fit_feature_transform(dataset.anchors[:2])
    x_train, y_train, a_train = _stack_anchors(dataset.anchors[:2], transform)
    x_val, y_val, a_val = _stack_anchors(dataset.anchors[2:], transform)
    _seed_everything(seed)
    model = FutureValueEncoder(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.encoder_lr)
    rng = np.random.default_rng(seed)
    positives, negatives = float(y_train.sum()), float(len(y_train) - y_train.sum())
    pos_weight = negatives / positives if positives > 0 and negatives > 0 else 1.0
    best_loss, best_epoch, best_state, bad = float("inf"), 1, None, 0
    for epoch in range(1, cfg.encoder_epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        for s in range(0, len(order), cfg.encoder_batch_size):
            idx = order[s : s + cfg.encoder_batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            ab = torch.as_tensor(a_train[idx], dtype=torch.float32, device=device)
            _, logit, pred = model(xb)
            loss = future_value_loss(logit, pred, yb, ab, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
            yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
            av = torch.as_tensor(a_val, dtype=torch.float32, device=device)
            _, logit, pred = model(xv)
            val_loss = float(future_value_loss(logit, pred, yv, av, pos_weight))
        if val_loss < best_loss - 1e-10:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
        if bad >= cfg.encoder_patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    diagnostics = _encoder_diagnostics(model, dataset.anchors[-1], transform, device)
    diagnostics.update(best_epoch=best_epoch, best_val_loss=best_loss)

    # Epoch choice is now fixed. Refit preprocessing and encoder on all train-internal anchors.
    final_transform = fit_feature_transform(dataset.anchors)
    x_all, y_all, a_all = _stack_anchors(dataset.anchors, final_transform)
    _seed_everything(seed)
    final_model = FutureValueEncoder(x_all.shape[1]).to(device)
    _train_encoder_epochs(
        final_model, x_all, y_all, a_all, best_epoch, cfg, seed, device
    )
    final_model.eval()
    n_users = dataset.n_users
    h_all = np.zeros((n_users, 16), np.float32)
    ev_all = np.zeros(n_users, np.float32)
    xs = torch.as_tensor(
        transform_features(final_snapshot, final_transform),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        h, logit, log_amount = final_model(xs)
        ev = torch.sigmoid(logit) * torch.expm1(log_amount)
    h_all[final_snapshot.user_ids] = h.cpu().numpy()
    ev_all[final_snapshot.user_ids] = ev.cpu().numpy()
    for p in final_model.parameters():
        p.requires_grad_(False)
    return EncoderArtifact(
        final_model, final_transform, best_epoch, diagnostics, h_all, ev_all
    )


class CLVResidualModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        h_clv: torch.Tensor,
        dim: int,
        constant: bool = False,
    ):
        super().__init__()
        for p in base_model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            up, ip, *_ = base_model.embeddings()
        self.register_buffer("base_user", up.detach().clone(), persistent=False)
        self.register_buffer("base_item", ip.detach().clone(), persistent=False)
        has_clv = h_clv.abs().sum(dim=1) > 0
        if constant:
            mean_h = (
                h_clv[has_clv].mean(0, keepdim=True) if has_clv.any() else h_clv[:1]
            )
            h_clv = mean_h.expand_as(h_clv).clone()
            h_clv[~has_clv] = 0.0
        self.register_buffer("h_clv", h_clv.detach().clone(), persistent=False)
        self.register_buffer("has_clv", has_clv.to(h_clv.dtype), persistent=False)
        self.adapter = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(), nn.Linear(32, dim), nn.Tanh()
        )
        self.gate_net = nn.Sequential(
            nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid()
        )
        nn.init.normal_(self.adapter[-2].weight, std=1e-3)
        nn.init.zeros_(self.adapter[-2].bias)
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.constant_(self.gate_net[-2].bias, -2.0)

    def residual(self):
        return self.adapter(self.h_clv)

    def gate(self):
        return self.gate_net(self.h_clv).squeeze(-1) * self.has_clv

    def residual_for(self, users):
        return self.adapter(self.h_clv[users])

    def gate_for(self, users):
        return self.gate_net(self.h_clv[users]).squeeze(-1) * self.has_clv[users]

    def embeddings(self, need_value=True):
        return self.base_user, self.base_item, self.residual(), self.base_item

    def bpr_loss(self, u, i, j, lam=1.0):
        r, g = self.residual_for(u), self.gate_for(u)
        zu = self.base_user[u] + lam * g[:, None] * r
        pos = (zu * self.base_item[i]).sum(1)
        neg = (zu * self.base_item[j]).sum(1)
        return -F.logsigmoid(pos - neg).mean()

    def trainable_parameters(self):
        return list(self.adapter.parameters()) + list(self.gate_net.parameters())


def residual_scores(base_user, base_item, residual, gate, lam: float):
    return (base_user + lam * gate.unsqueeze(-1) * residual) @ base_item.T


@torch.no_grad()
def assert_lambda_zero_equivalence(model: CLVResidualModel, n_check: int = 256):
    """Check global finiteness and deterministic sampled scores/Top-K.

    Materializing every H&M user-item score would add another full evaluation.
    Finite residual/gate values plus the literal zero multiplier establish the
    global invariant; the sampled matrix exercises the numerical and Top-K path.
    """
    if (
        not torch.isfinite(model.residual()).all()
        or not torch.isfinite(model.gate()).all()
    ):
        raise RuntimeError(
            "residual 또는 gate가 유한하지 않아 lambda=0 불변식을 보장할 수 없습니다"
        )
    n_users = model.base_user.shape[0]
    users = torch.linspace(
        0, n_users - 1, min(n_users, n_check), device=model.base_user.device
    ).long()
    residual, gate = model.residual_for(users), model.gate_for(users)
    max_k = min(50, model.base_item.shape[0])
    base = model.base_user[users] @ model.base_item.T
    zero = residual_scores(model.base_user[users], model.base_item, residual, gate, 0.0)
    if not torch.equal(base, zero):
        raise RuntimeError("lambda=0 점수가 M1과 수치적으로 동일하지 않습니다")
    if not torch.equal(
        base.topk(max_k, dim=1).indices, zero.topk(max_k, dim=1).indices
    ):
        raise RuntimeError("lambda=0 Top-K가 M1과 동일하지 않습니다")


def train_residual_adapter(
    model: CLVResidualModel,
    base_model: nn.Module,
    data: dict,
    base_cfg: dict,
    cfg: ResidualConfig,
    seed: int,
    eval_recall,
) -> dict:
    """Train only adapter/gate with the same rows and uniform BPR sampler as M1."""
    import lightgcn_clv_v3 as v3

    _seed_everything(seed)
    before = state_hash(base_model)
    params = model.trainable_parameters()
    opt = torch.optim.Adam(params, lr=cfg.adapter_lr)
    rng = np.random.default_rng(seed)
    tr_u, tr_i = data["tr_u"], data["tr_i"]
    n = len(tr_u)
    batch = int(base_cfg["BATCH_SIZE"])
    best, best_state, best_epoch, bad = -float("inf"), None, 0, 0
    updates = samples = 0
    last_loss = float("nan")
    started = time.time()
    for epoch in range(1, cfg.adapter_epochs + 1):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s : s + batch]
            bu, bi = tr_u[idx], tr_i[idx]
            bj = v3.sample_negatives(
                bu,
                bi,
                data["n_items"],
                data["pos_key"],
                rng,
                base_cfg["NEG_MODE"],
                data["item_cat"],
                data["cat_items"],
            )
            device = model.base_user.device
            loss = model.bpr_loss(
                torch.as_tensor(bu, dtype=torch.long, device=device),
                torch.as_tensor(bi, dtype=torch.long, device=device),
                torch.as_tensor(bj, dtype=torch.long, device=device),
                lam=1.0,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            updates += 1
            samples += len(idx)
        score = float(eval_recall(model))
        if score > best + 1e-12:
            best, best_epoch, bad = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
        if bad >= cfg.adapter_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    after = state_hash(base_model)
    if before != after:
        raise RuntimeError("adapter 학습 중 동결된 M1 state가 변경됐습니다")
    return {
        "loss": "plain_bpr",
        "best_epoch": best_epoch,
        "best_val_recall@10": best,
        "updates": updates,
        "samples": samples,
        "last_loss": last_loss,
        "wall_clock_sec": time.time() - started,
        "base_hash_before": before,
        "base_hash_after": after,
    }


def normalize_flat_metrics(flat: dict) -> dict:
    out = {}
    for key, value in flat.items():
        if key.startswith("entropy@"):
            key = "exposure_" + key
        out[key] = value
    return out


def select_lambda(rows: Sequence[dict], baseline: dict, tolerance: float = 0.01):
    table = pd.DataFrame(rows).copy()
    guards = [f"{m}@{k}" for m in ("recall", "ndcg") for k in (10, 20, 50)]
    eligible = np.ones(len(table), dtype=bool)
    for key in guards:
        if key not in table or key not in baseline:
            raise KeyError(f"lambda 선택에 필요한 {key}가 없습니다")
        eligible &= table[key].to_numpy() >= float(baseline[key]) * (1.0 - tolerance)
    table["eligible"] = eligible
    candidates = table[table.eligible]
    if len(candidates) == 0:
        return 0.0, table
    best_econ = candidates["revenue@10"].max()
    chosen = candidates[np.isclose(candidates["revenue@10"], best_econ)]
    return float(chosen["lambda"].min()), table


def configure_residual_run(dataset: str, **overrides) -> ResidualConfig:
    if dataset not in ("hm", "dunnhumby"):
        raise ValueError("dataset은 hm 또는 dunnhumby여야 합니다")
    cfg = ResidualConfig(dataset=dataset)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"알 수 없는 residual 설정: {key}")
        setattr(cfg, key, value)
    if cfg.eval_holdout and not cfg.eval_test:
        raise ValueError("holdout 확증 전에 test 확증 설정을 명시해야 합니다")
    return cfg


def validate_result_metrics(flat: dict, ks: Iterable[int] = (10, 20, 50)):
    required = []
    for k in ks:
        required += [
            f"recall@{k}",
            f"ndcg@{k}",
            f"revenue@{k}",
            f"coverage@{k}",
            f"n_distinct@{k}",
            f"exposure_entropy@{k}",
            f"eff_catalog@{k}",
            f"top10_share@{k}",
            f"top100_share@{k}",
        ]
    missing = [key for key in required if key not in flat]
    if missing:
        raise KeyError(f"필수 결과지표 누락: {', '.join(missing)}")


def state_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _effective_score_ratio(
    model: CLVResidualModel, seed: int, n_sample: int = 256
) -> dict:
    rng = np.random.default_rng(seed)
    n = model.base_user.shape[0]
    users = torch.as_tensor(
        rng.choice(n, min(n, n_sample), replace=False),
        dtype=torch.long,
        device=model.base_user.device,
    )
    with torch.no_grad():
        pref = model.base_user[users] @ model.base_item.T
        residual = (
            model.gate_for(users)[:, None] * model.residual_for(users)
        ) @ model.base_item.T
    return {
        "std_preference_score": float(pref.std()),
        "std_effective_residual_score": float(residual.std()),
        "effective_score_ratio": float(residual.std() / (pref.std() + 1e-12)),
        "gate_mean": float(model.gate_for(users).mean()),
        "residual_norm_mean": float(model.residual_for(users).norm(dim=1).mean()),
    }


def _input_manifest_hash(input_manifest: dict) -> str:
    identity = {
        label: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        for label, entry in input_manifest.items()
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()


def _provenance_payload(
    cfg: ResidualConfig,
    base_cfg: dict,
    input_manifest: dict,
    source_revision: str,
    baseline_state_hashes: dict[str, str],
) -> dict:
    return {
        "residual": asdict(cfg),
        "base": {
            k: base_cfg[k]
            for k in (
                "DIM",
                "N_LAYERS",
                "BATCH_SIZE",
                "LR",
                "EPOCHS",
                "EARLY_STOP",
                "WINDOW_DAYS",
                "VAL_DAYS",
                "TEST_DAYS",
                "HOLDOUT_DAYS",
                "MIN_USER_INTER",
                "MIN_ITEM_INTER",
                "NEG_MODE",
                "GRAPH_MODE",
                "LOSS_MODE",
            )
        },
        "code": CODE_VERSION,
        "features": NUMERIC_FEATURES,
        "source_revision": source_revision,
        "input_manifest_hash": _input_manifest_hash(input_manifest),
        "baseline_state_hashes": baseline_state_hashes,
    }


def _result_fingerprint(
    cfg: ResidualConfig,
    base_cfg: dict,
    input_manifest: dict,
    source_revision: str,
    baseline_state_hashes: dict[str, str],
) -> str:
    payload = _provenance_payload(
        cfg,
        base_cfg,
        input_manifest,
        source_revision,
        baseline_state_hashes,
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _checkpoint_fingerprint(
    cfg: ResidualConfig,
    base_cfg: dict,
    input_manifest: dict,
    source_revision: str,
    seed: int,
    baseline_state_hash: str,
) -> str:
    return _result_fingerprint(
        cfg,
        base_cfg,
        input_manifest,
        source_revision,
        {str(seed): baseline_state_hash},
    )


def _checkpoint_path(
    out_dir: str | Path,
    artifact_name: str,
    dataset: str,
    seed: int,
    fingerprint: str,
) -> Path:
    return Path(out_dir) / f"{artifact_name}_{dataset}_s{seed}_{fingerprint}.pt"


def _checkpoint_provenance(
    cfg: ResidualConfig,
    base_cfg: dict,
    input_manifest: dict,
    source_revision: str,
    seed: int,
    baseline_state_hash: str,
    m1_checkpoint: str | Path,
) -> dict:
    fingerprint = _checkpoint_fingerprint(
        cfg,
        base_cfg,
        input_manifest,
        source_revision,
        seed,
        baseline_state_hash,
    )
    return {
        "checkpoint_fingerprint": fingerprint,
        "source_revision": source_revision,
        "input_manifest": input_manifest,
        "input_manifest_hash": _input_manifest_hash(input_manifest),
        "baseline_state_hash": baseline_state_hash,
        "seed": int(seed),
        "m1_checkpoint": str(m1_checkpoint),
        "config": asdict(cfg),
    }


def _save_provenance_checkpoint(
    path: str | Path,
    payload: dict,
    provenance: dict,
) -> Path:
    path = Path(path)
    torch.save(payload | {"provenance": provenance}, path)
    return path


def preflight_summary(cfg: ResidualConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seed_list),
        "window": "full official train (~2 years)",
        "encoder_windows": {
            "input_days": cfg.input_days,
            "target_days": cfg.target_days,
            "anchor_offsets": list(cfg.anchor_offsets),
        },
        "lambda_eval": list(cfg.lambda_eval),
        "selection": "six Recall/NDCG@10/20/50 >= 99% of M1, then max weighted-hit@10",
        "test_enabled": cfg.eval_test,
        "holdout_enabled": cfg.eval_holdout,
        "models": ["m1", "clv_residual"]
        + (["constant_control"] if cfg.include_constant_control else []),
    }


def run_experiment(cfg: ResidualConfig | None = None) -> pd.DataFrame:
    """Run the approved screening/confirmation pipeline.

    The default is Dunnhumby, seed 42, validation only.  Test labels are not
    constructed unless ``cfg.eval_test`` is explicitly true.
    """
    import lightgcn_clv_moe as moe
    import lightgcn_clv_v3 as v3

    cfg = cfg or configure_residual_run("dunnhumby")
    input_manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_id = moe.manifest_hash(input_manifest)
    revision = moe.source_revision()
    out_dir = Path(cfg.out_dir or (v3.default_out_dir(cfg.dataset) + "_clv_residual"))
    out_dir.mkdir(parents=True, exist_ok=True)
    m1_root = Path(cfg.m1_checkpoint_dir or v3.default_out_dir(cfg.dataset))
    m1_dir = m1_root / f"data_{input_id[:12]}"
    base_cfg = v3.configure_run(
        dataset=cfg.dataset,
        out_dir=str(m1_dir),
        ARCH="pref_only",
        SEED_LIST=list(cfg.seed_list),
        WINDOW_DAYS=None,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        GATE_MODE="clv",
        NEG_MODE="uniform",
        EVAL_TEST=cfg.eval_test,
        EVAL_HOLDOUT=cfg.eval_holdout,
    )
    if (
        base_cfg["ARCH"] != "pref_only"
        or base_cfg["GRAPH_MODE"] != "binary"
        or base_cfg["LOSS_MODE"] != "plain"
    ):
        raise RuntimeError(
            "CLV-Residual M2의 기준은 반드시 순수 M1(pref_only/binary/plain)이어야 합니다"
        )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    data = v3.prepare_data(base_cfg, v3.DCFG)
    anchors = build_anchor_examples(
        data["train"],
        data["n_users"],
        v3.DCFG["is_date"],
        cfg.input_days,
        cfg.target_days,
        cfg.anchor_offsets,
    )
    snapshot = build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    ones_gate = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    model_variants = [("clv_residual", False)]
    if cfg.include_constant_control:
        model_variants.append(("constant_control", True))

    rows, encoder_records, train_records = [], {}, {}
    val_per_user, base_per_user, checkpoint_paths = {}, {}, {}
    artifact_checkpoint_paths = {}
    baseline_state_hashes = {}
    checkpoint_fingerprints = {}
    baseline_rows_by_seed = {}
    for seed in cfg.seed_list:
        artifact = train_future_value_encoder(anchors, snapshot, cfg, seed, v3.DEVICE)
        encoder_records[str(seed)] = artifact.diagnostics
        seg_th = v3.segment_thresholds(artifact.ev_all, base_cfg["SEG_EDGES"])
        caches = {
            name: v3.EvalCache(gt, rev, artifact.ev_all, seg_th, data["n_items"])
            for name, (gt, rev) in data["splits"].items()
        }
        m1_checkpoint = Path(base_cfg["OUT_DIR"]) / (
            f"ckpt_pref_only_{cfg.dataset}_s{seed}_"
            f"{v3.cfg_hash(base_cfg, v3.DCFG, 'pref_only', seed)}.pt"
        )
        m1_existed_before = m1_checkpoint.exists()
        base_model, _ = v3.get_or_train(
            "pref_only",
            seed,
            data,
            ones_gate,
            data["x_val_u"],
            x_item,
            item_cat,
            meta,
            caches["val"],
            base_cfg,
        )
        base_model.eval()
        baseline_state_hash = state_hash(base_model)
        baseline_state_hashes[str(seed)] = baseline_state_hash
        if not m1_checkpoint.exists():
            raise RuntimeError(f"M1 checkpoint was not saved: {m1_checkpoint}")
        moe.validate_or_write_m1_manifest(
            m1_checkpoint,
            input_manifest,
            config_hash=v3.cfg_hash(base_cfg, v3.DCFG, "pref_only", seed),
            state_hash_value=baseline_state_hash,
            existed_before=m1_existed_before,
        )
        checkpoint_provenance = _checkpoint_provenance(
            cfg,
            base_cfg,
            input_manifest,
            revision,
            seed,
            baseline_state_hash,
            m1_checkpoint,
        )
        checkpoint_fingerprint = checkpoint_provenance["checkpoint_fingerprint"]
        checkpoint_fingerprints[str(seed)] = checkpoint_fingerprint
        encoder_path = _checkpoint_path(
            out_dir, "encoder", cfg.dataset, seed, checkpoint_fingerprint
        )
        _save_provenance_checkpoint(
            encoder_path,
            {
                "state": artifact.model.state_dict(),
                "transform": {
                    "mean": artifact.transform.mean,
                    "std": artifact.transform.std,
                    "feature_names": artifact.transform.feature_names,
                },
                "h_all": artifact.h_all,
                "ev_all": artifact.ev_all,
                "best_epoch": artifact.best_epoch,
                "diagnostics": artifact.diagnostics,
            },
            checkpoint_provenance,
        )
        artifact_checkpoint_paths[f"encoder_s{seed}"] = str(encoder_path)
        artifact_checkpoint_paths[f"m1_s{seed}"] = str(m1_checkpoint)
        base_eval = v3.evaluate(
            base_model,
            0.0,
            ones_gate,
            caches["val"],
            meta,
            base_cfg["K_LIST"],
            data["csr_ptr"],
            data["csr_items"],
            base_cfg,
            per_user=True,
        )
        base_per_user[seed] = base_eval.pop("per_user")
        base_flat = normalize_flat_metrics(v3.flatten(base_eval))
        validate_result_metrics(base_flat, base_cfg["K_LIST"])
        baseline_rows_by_seed[seed] = base_flat
        rows.append(
            {
                "seed": seed,
                "model_id": "m1",
                "split": "val",
                "lambda": 0.0,
                "role": "baseline",
                **base_flat,
            }
        )

        for model_id, constant in model_variants:
            h = torch.as_tensor(artifact.h_all, dtype=torch.float32, device=v3.DEVICE)
            model = CLVResidualModel(
                base_model, h, base_cfg["DIM"], constant=constant
            ).to(v3.DEVICE)
            before_encoder = state_hash(artifact.model)

            def eval_recall(candidate):
                return v3.evaluate(
                    candidate,
                    1.0,
                    candidate.gate().detach(),
                    caches["val"],
                    meta,
                    [10],
                    data["csr_ptr"],
                    data["csr_items"],
                    base_cfg,
                )["overall"][10]["recall"]

            stats = train_residual_adapter(
                model, base_model, data, base_cfg, cfg, seed, eval_recall
            )
            assert_lambda_zero_equivalence(model)
            if state_hash(artifact.model) != before_encoder:
                raise RuntimeError("adapter 학습 중 동결된 CLV encoder가 변경됐습니다")
            train_records[f"{model_id}_s{seed}"] = {
                **stats,
                **_effective_score_ratio(model, seed),
                "encoder_hash": before_encoder,
                "adapter_parameters": sum(
                    p.numel() for p in model.trainable_parameters()
                ),
            }
            ckpt = _checkpoint_path(
                out_dir,
                f"adapter_{model_id}",
                cfg.dataset,
                seed,
                checkpoint_fingerprint,
            )
            _save_provenance_checkpoint(
                ckpt,
                {
                    "state": model.state_dict(),
                    "h_all": artifact.h_all,
                    "ev_all": artifact.ev_all,
                    "encoder_path": str(encoder_path),
                    "train_stats": train_records[f"{model_id}_s{seed}"],
                },
                checkpoint_provenance,
            )
            checkpoint_paths[(model_id, seed)] = ckpt
            val_per_user[(model_id, seed)] = {}
            for lam in cfg.lambda_eval:
                result = v3.evaluate(
                    model,
                    lam,
                    model.gate().detach(),
                    caches["val"],
                    meta,
                    base_cfg["K_LIST"],
                    data["csr_ptr"],
                    data["csr_items"],
                    base_cfg,
                    per_user=True,
                )
                val_per_user[(model_id, seed)][lam] = result.pop("per_user")
                flat = normalize_flat_metrics(v3.flatten(result))
                validate_result_metrics(flat, base_cfg["K_LIST"])
                rows.append(
                    {
                        "seed": seed,
                        "model_id": model_id,
                        "split": "val",
                        "lambda": lam,
                        "role": "model",
                        **flat,
                    }
                )
            del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    selected, selection_tables = {}, {}
    base_mean = {
        k: float(np.mean([r[k] for r in baseline_rows_by_seed.values()]))
        for k in next(iter(baseline_rows_by_seed.values()))
    }
    for model_id, _ in model_variants:
        mean_rows = (
            frame[(frame.model_id == model_id) & (frame.split == "val")]
            .groupby("lambda", as_index=False)
            .mean(numeric_only=True)
            .to_dict("records")
        )
        lam, table = select_lambda(mean_rows, base_mean, cfg.accuracy_tolerance)
        selected[model_id] = lam
        selection_tables[model_id] = table.to_dict("records")

    delta_records = []
    for model_id, _ in model_variants:
        lam = selected[model_id]
        for metric in ("recall", "ndcg", "revenue", "arp"):
            diffs = [
                val_per_user[(model_id, seed)][lam][metric]
                - base_per_user[seed][metric]
                for seed in cfg.seed_list
            ]
            ci = v3.paired_bootstrap(diffs, base_cfg["N_BOOT"])
            delta_records.append(
                {
                    "model_id": model_id,
                    "split": "val",
                    "lambda": lam,
                    "metric": metric,
                    **ci,
                }
            )

    # Protected confirmation: only the validation-selected lambda is evaluated on test.
    if cfg.eval_test:
        test_base_per_user, test_model_per_user = {}, {}
        for seed in cfg.seed_list:
            blob_any = torch.load(
                checkpoint_paths[(model_variants[0][0], seed)],
                map_location=v3.DEVICE,
                weights_only=False,
            )
            ev_all = np.asarray(blob_any["ev_all"], np.float32)
            seg_th = v3.segment_thresholds(ev_all, base_cfg["SEG_EDGES"])
            test_cache = v3.EvalCache(
                *data["splits"]["test"], ev_all, seg_th, data["n_items"]
            )
            base_model, _ = v3.get_or_train(
                "pref_only",
                seed,
                data,
                ones_gate,
                data["x_val_u"],
                x_item,
                item_cat,
                meta,
                test_cache,
                base_cfg,
            )
            base_test = v3.evaluate(
                base_model,
                0.0,
                ones_gate,
                test_cache,
                meta,
                base_cfg["K_LIST"],
                data["csr_ptr"],
                data["csr_items"],
                base_cfg,
                per_user=True,
            )
            base_test_pu = base_test.pop("per_user")
            test_base_per_user[seed] = base_test_pu
            base_test_flat = normalize_flat_metrics(v3.flatten(base_test))
            rows.append(
                {
                    "seed": seed,
                    "model_id": "m1",
                    "split": "test",
                    "lambda": 0.0,
                    "role": "baseline",
                    **base_test_flat,
                }
            )
            for model_id, constant in model_variants:
                blob = torch.load(
                    checkpoint_paths[(model_id, seed)],
                    map_location=v3.DEVICE,
                    weights_only=False,
                )
                h = torch.as_tensor(
                    blob["h_all"], dtype=torch.float32, device=v3.DEVICE
                )
                model = CLVResidualModel(
                    base_model, h, base_cfg["DIM"], constant=constant
                ).to(v3.DEVICE)
                model.load_state_dict(blob["state"])
                lam = selected[model_id]
                result = v3.evaluate(
                    model,
                    lam,
                    model.gate().detach(),
                    test_cache,
                    meta,
                    base_cfg["K_LIST"],
                    data["csr_ptr"],
                    data["csr_items"],
                    base_cfg,
                    per_user=True,
                )
                model_test_pu = result.pop("per_user")
                test_model_per_user[(model_id, seed)] = model_test_pu
                flat = normalize_flat_metrics(v3.flatten(result))
                validate_result_metrics(flat)
                rows.append(
                    {
                        "seed": seed,
                        "model_id": model_id,
                        "split": "test",
                        "lambda": lam,
                        "role": "primary",
                        **flat,
                    }
                )
            del base_model
        for model_id, _ in model_variants:
            lam = selected[model_id]
            for metric in ("recall", "ndcg", "revenue", "arp"):
                diffs = [
                    test_model_per_user[(model_id, seed)][metric]
                    - test_base_per_user[seed][metric]
                    for seed in cfg.seed_list
                ]
                ci = v3.paired_bootstrap(diffs, base_cfg["N_BOOT"])
                delta_records.append(
                    {
                        "model_id": model_id,
                        "split": "test",
                        "lambda": lam,
                        "metric": metric,
                        **ci,
                    }
                )

    frame = pd.DataFrame(rows)
    fingerprint = _result_fingerprint(
        cfg,
        base_cfg,
        input_manifest,
        revision,
        baseline_state_hashes,
    )
    stem = f"clv_residual_{cfg.dataset}_{fingerprint}"
    result_csv = out_dir / f"{stem}.csv"
    delta_csv = out_dir / f"{stem}_delta.csv"
    result_json = out_dir / f"{stem}.json"
    frame.to_csv(result_csv, index=False, float_format="%.8f")
    pd.DataFrame(delta_records).to_csv(delta_csv, index=False)
    checkpoint_paths_json = {
        f"{model_id}_s{seed}": str(path)
        for (model_id, seed), path in checkpoint_paths.items()
    } | artifact_checkpoint_paths
    checkpoint_sha256 = {
        key: moe.file_sha256(path)
        for key, path in checkpoint_paths_json.items()
        if Path(path).is_file()
    }
    frame.attrs["result_fingerprint"] = fingerprint
    frame.attrs["result_paths"] = {
        "csv": str(result_csv),
        "delta_csv": str(delta_csv),
        "json": str(result_json),
    }
    with result_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "code_version": CODE_VERSION,
                "source_revision": revision,
                "result_fingerprint": fingerprint,
                "input_manifest": input_manifest,
                "provenance": _provenance_payload(
                    cfg,
                    base_cfg,
                    input_manifest,
                    revision,
                    baseline_state_hashes,
                ),
                "config": asdict(cfg),
                "base_config": {k: v for k, v in base_cfg.items() if k != "OUT_DIR"},
                "baseline_state_hashes": baseline_state_hashes,
                "checkpoint_fingerprints": checkpoint_fingerprints,
                "checkpoint_paths": checkpoint_paths_json,
                "checkpoint_sha256": checkpoint_sha256,
                "data_stats": data.get("data_stats", {}),
                "selected_lambda": selected,
                "selection_tables": selection_tables,
                "encoder_diagnostics": encoder_records,
                "training": train_records,
                "absolute_rows": frame.to_dict("records"),
                "delta": delta_records,
                "interpretation": {
                    "ev": "90-day expected purchase value, not lifetime CLV",
                    "revenue": "price/purchase-amount weighted hit, not incremental revenue",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"저장: {result_csv}")
    print(f"validation 선택 λ: {selected}")
    return frame


if __name__ == "__main__":
    run_experiment()
