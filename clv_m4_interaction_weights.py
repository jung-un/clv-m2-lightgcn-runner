"""Train-only sample weights for CLV-conditioned M4 BPR experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


@dataclass(frozen=True)
class M4InteractionWeights:
    row_weights: np.ndarray
    row_signal: np.ndarray
    q_clv: np.ndarray
    row_item_contribution_rank: np.ndarray
    diagnostics: dict


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / len(values)


def build_m4_interaction_weights(train, clv, *, mode, beta):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta는 유한한 양수여야 합니다")
    required = {"u_idx", "i_idx", "v"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"M4 interaction weight 필수 컬럼 누락: {', '.join(missing)}")
    if train.empty:
        raise ValueError("M4 interaction weight에 사용할 train 행이 없습니다")
    if not np.isfinite(train["v"].to_numpy(np.float64)).all():
        raise ValueError("train 거래가치에 비유한 값이 있습니다")
    basket_key = "b_raw" if "b_raw" in train.columns else "t"
    if basket_key not in train.columns:
        raise ValueError("M4 interaction weight는 b_raw 또는 t 컬럼이 필요합니다")

    users = train["u_idx"].to_numpy(np.int64)
    clv = np.asarray(clv, dtype=np.float64).copy()
    if users.min() < 0 or users.max() >= len(clv):
        raise ValueError("u_idx가 CLV 배열 범위를 벗어났습니다")
    if np.all(np.isnan(clv)):
        raise ValueError("CLV가 전부 NaN이라 M4 가중치를 만들 수 없습니다")
    clv[np.isnan(clv)] = np.nanmedian(clv)
    q_clv = _percentile(clv)

    basket = (
        train.groupby(["u_idx", basket_key], sort=False)["v"]
        .sum()
        .rename("basket_value")
        .reset_index()
    )
    line = (
        train.groupby(["u_idx", "i_idx", basket_key], sort=False)["v"]
        .sum()
        .rename("line_value")
        .reset_index()
        .merge(
            basket,
            on=["u_idx", basket_key],
            how="left",
            validate="many_to_one",
        )
    )
    valid = line["basket_value"].to_numpy(np.float64) > 0
    line["share"] = 0.0
    line.loc[valid, "share"] = np.clip(
        line.loc[valid, "line_value"].to_numpy(np.float64)
        / line.loc[valid, "basket_value"].to_numpy(np.float64),
        0.0,
        None,
    )
    pair = (
        line.groupby(["u_idx", "i_idx"], sort=True)["share"]
        .mean()
        .rename("contribution")
        .reset_index()
    )
    rank = pair.groupby("u_idx")["contribution"].rank(method="average")
    count = pair.groupby("u_idx")["contribution"].transform("size")
    pair["q_item"] = (rank - 0.5) / count
    pair_index = pair.set_index(["u_idx", "i_idx"])
    row_index = pd.MultiIndex.from_arrays(
        [train["u_idx"].to_numpy(), train["i_idx"].to_numpy()]
    )
    q_item = pair_index["q_item"].reindex(row_index).to_numpy(np.float64)

    if mode == "pair_contribution":
        signal = q_item
    elif mode == "clv_pair":
        signal = q_clv[users] * q_item
    else:
        raise ValueError(
            f"mode={mode!r} — pair_contribution|clv_pair 중 하나여야 합니다"
        )
    z = 2.0 * _percentile(signal) - 1.0
    raw = np.exp(float(beta) * z)
    weights = raw / raw.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("M4 interaction weight가 유한한 양수가 아닙니다")
    correlation = spearmanr(clv[users], weights).correlation
    diagnostics = {
        "mode": mode,
        "beta": float(beta),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_min": float(weights.min()),
        "weight_median": float(np.median(weights)),
        "weight_max": float(weights.max()),
        "clv_weight_spearman": (
            0.0 if np.isnan(correlation) else float(correlation)
        ),
        "nonpositive_basket_line_share": float((~valid).mean()),
    }
    return M4InteractionWeights(
        row_weights=weights.astype(np.float32),
        row_signal=signal.astype(np.float32),
        q_clv=q_clv.astype(np.float32),
        row_item_contribution_rank=q_item.astype(np.float32),
        diagnostics=diagnostics,
    )
