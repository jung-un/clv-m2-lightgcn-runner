"""Checkpoint-only diagnostic for CLV-conditioned negative sampling.

No training, checkpoint selection, reranking, test or holdout. The proposed M4'
keeps the original single-negative BPR and changes only where that negative
comes from: a low-CLV customer keeps the uniform draw, while a high-CLV
customer's negative is the highest scoring of several uniform candidates. That
design assumes two things about the existing M1 checkpoint:

1. saturation - for high-CLV customers a uniformly drawn unseen item is already
   ranked far below their purchased items, so the comparison carries little
   learning signal;
2. headroom - replacing that draw with the hardest of several candidates adds
   more signal for high-CLV customers than for low-CLV ones.

The learning signal of one (positive, negative) pair is the BPR gradient
factor ``sigmoid(s(u,j) - s(u,i+))``: it is near zero once the pair is already
ordered by a wide margin. Both quantities are measured per user and summarized
by the fixed CLV segments used everywhere else in this project.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_candidate_relation_diagnostic as relation
import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_m5_value_precision_diagnostic as precision
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-negative-saturation-diagnostic-v1"
POSITIVE_SAMPLES = 20
CANDIDATES = 9
SAMPLING_SEED = 42
BOOTSTRAP_SAMPLES = 2000


def configure_negative_saturation_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> relation.CandidateRelationDiagnosticConfig:
    dataset = dataset.lower()
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir(dataset)}_m4_negative_saturation_diagnostic_v1"
        )
    }
    return relation.configure_candidate_relation_diagnostic(
        dataset, **(defaults | overrides)
    )


def preflight_summary(cfg) -> dict:
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "training": False,
        "checkpoint_selection": False,
        "final_test_executed": False,
        "holdout_executed": False,
        "model": "existing seed-42 M1 checkpoint (one uniform negative BPR)",
        "sampling": {
            "positives_per_user": POSITIVE_SAMPLES,
            "candidates_per_positive": CANDIDATES,
            "negative_pool": "items the user did not buy in the training window",
            "seed": SAMPLING_SEED,
        },
        "quantities": {
            "uniform_signal": (
                "mean sigmoid(s(u,j) - s(u,i+)) over sampled positives with one "
                "uniform negative each; the BPR gradient factor"
            ),
            "hard_signal": (
                "the same with the highest scoring of "
                f"{CANDIDATES} uniform candidates as the negative"
            ),
            "headroom": "hard_signal - uniform_signal",
            "p_correct_uniform": "share of sampled pairs already ordered correctly",
        },
        "reading_rule": {
            "saturation": (
                "fixed high-CLV uniform_signal < fixed low-CLV uniform_signal and "
                "the user-level 95% bootstrap interval of that difference excludes zero"
            ),
            "headroom": (
                "fixed high-CLV headroom > fixed low-CLV headroom with the same "
                "interval condition"
            ),
            "design_supported": "saturation and headroom, in both datasets",
        },
        "statistical_note": (
            "single-checkpoint descriptive development diagnostic; bootstrap "
            "intervals describe this evaluation population only and no "
            "generalization or causal claim is made"
        ),
        "out_dir": cfg.out_dir,
    }


def sample_user_positives(
    users: np.ndarray,
    csr_ptr: np.ndarray,
    csr_items: np.ndarray,
    rng: np.random.Generator,
    *,
    per_user: int = POSITIVE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw up to ``per_user`` purchased items for each evaluation user."""

    sampled_users, sampled_items = [], []
    for user in np.asarray(users, dtype=np.int64):
        left, right = int(csr_ptr[user]), int(csr_ptr[user + 1])
        owned = csr_items[left:right]
        if not len(owned):
            continue
        take = owned if len(owned) <= per_user else rng.choice(owned, per_user, replace=False)
        sampled_users.append(np.full(len(take), user, dtype=np.int64))
        sampled_items.append(np.asarray(take, dtype=np.int64))
    if not sampled_users:
        raise RuntimeError("학습 양성이 있는 평가 사용자가 없습니다")
    return np.concatenate(sampled_users), np.concatenate(sampled_items)


@torch.no_grad()
def pair_signals(
    user_embedding: torch.Tensor,
    item_embedding: torch.Tensor,
    users: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    batch_size: int = 65536,
) -> dict[str, np.ndarray]:
    """BPR gradient factors for a uniform negative and for the hardest candidate."""

    uniform, hard, correct = [], [], []
    for start in range(0, len(users), batch_size):
        stop = start + batch_size
        user_rows = torch.as_tensor(
            users[start:stop], dtype=torch.long, device=user_embedding.device
        )
        positive_rows = torch.as_tensor(
            positives[start:stop], dtype=torch.long, device=user_embedding.device
        )
        negative_rows = torch.as_tensor(
            negatives[start:stop], dtype=torch.long, device=user_embedding.device
        )
        user_vectors = user_embedding.index_select(0, user_rows)
        positive_scores = (user_vectors * item_embedding[positive_rows]).sum(dim=1)
        negative_scores = (
            user_vectors[:, None, :] * item_embedding[negative_rows]
        ).sum(dim=2)
        gap_uniform = negative_scores[:, 0] - positive_scores
        gap_hard = negative_scores.max(dim=1).values - positive_scores
        uniform.append(torch.sigmoid(gap_uniform).cpu().numpy())
        hard.append(torch.sigmoid(gap_hard).cpu().numpy())
        correct.append((gap_uniform < 0).float().cpu().numpy())
    return {
        "uniform_signal": np.concatenate(uniform).astype(np.float64),
        "hard_signal": np.concatenate(hard).astype(np.float64),
        "correct": np.concatenate(correct).astype(np.float64),
    }


def per_user_table(
    users: np.ndarray, signals: dict[str, np.ndarray], membership: pd.DataFrame
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "user_idx": np.asarray(users, dtype=np.int64),
            "uniform_signal": signals["uniform_signal"],
            "hard_signal": signals["hard_signal"],
            "p_correct_uniform": signals["correct"],
        }
    )
    grouped = frame.groupby("user_idx", sort=True).mean().reset_index()
    grouped["headroom"] = grouped.hard_signal - grouped.uniform_signal
    segments = membership.set_index("user_idx")["fixed_clv_segment"]
    grouped["fixed_clv_segment"] = grouped.user_idx.map(segments)
    return grouped.dropna(subset=["fixed_clv_segment"])


def summarize(per_user: pd.DataFrame) -> pd.DataFrame:
    columns = ["uniform_signal", "hard_signal", "headroom", "p_correct_uniform"]
    rows = [
        {
            "group_type": "overall",
            "group": "전체",
            "n_users": int(len(per_user)),
            **{column: float(per_user[column].mean()) for column in columns},
        }
    ]
    for name in fixed.SEGMENT_ORDER:
        selected = per_user[per_user.fixed_clv_segment.eq(name)]
        if selected.empty:
            continue
        rows.append(
            {
                "group_type": "fixed_clv_segment",
                "group": name,
                "n_users": int(len(selected)),
                **{column: float(selected[column].mean()) for column in columns},
            }
        )
    return pd.DataFrame(rows)


def bootstrap_high_minus_low(
    per_user: pd.DataFrame,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SAMPLING_SEED,
) -> dict:
    low_name, high_name = fixed.SEGMENT_ORDER[0], fixed.SEGMENT_ORDER[2]
    low = per_user[per_user.fixed_clv_segment.eq(low_name)]
    high = per_user[per_user.fixed_clv_segment.eq(high_name)]
    if low.empty or high.empty:
        raise RuntimeError("저CLV 또는 고CLV 사용자가 없습니다")

    rng = np.random.default_rng(seed)
    report = {}
    for column in ("uniform_signal", "headroom"):
        low_values = low[column].to_numpy(np.float64)
        high_values = high[column].to_numpy(np.float64)
        observed = float(high_values.mean() - low_values.mean())
        draws = np.empty(int(samples), dtype=np.float64)
        for index in range(int(samples)):
            draws[index] = (
                high_values[rng.integers(0, len(high_values), len(high_values))].mean()
                - low_values[rng.integers(0, len(low_values), len(low_values))].mean()
            )
        low_edge, high_edge = precision._percentile_interval(draws)
        report[column] = {
            "observed_high_minus_low": observed,
            "ci_low": low_edge,
            "ci_high": high_edge,
            "excludes_zero": bool(low_edge > 0.0 or high_edge < 0.0),
        }
    return {
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "low_clv_users": int(len(low)),
        "high_clv_users": int(len(high)),
        "contrasts": report,
    }


def dataset_reading(summary: pd.DataFrame, bootstrap: dict) -> dict:
    indexed = summary.set_index("group")
    low_name, high_name = fixed.SEGMENT_ORDER[0], fixed.SEGMENT_ORDER[2]
    uniform = bootstrap["contrasts"]["uniform_signal"]
    headroom = bootstrap["contrasts"]["headroom"]
    saturation = bool(
        uniform["observed_high_minus_low"] < 0.0 and uniform["excludes_zero"]
    )
    more_headroom = bool(
        headroom["observed_high_minus_low"] > 0.0 and headroom["excludes_zero"]
    )
    return {
        "low_clv_uniform_signal": float(indexed.at[low_name, "uniform_signal"]),
        "high_clv_uniform_signal": float(indexed.at[high_name, "uniform_signal"]),
        "low_clv_headroom": float(indexed.at[low_name, "headroom"]),
        "high_clv_headroom": float(indexed.at[high_name, "headroom"]),
        "saturation_in_high_clv": saturation,
        "more_headroom_in_high_clv": more_headroom,
        "design_supported_in_this_dataset": bool(saturation and more_headroom),
        "cross_dataset_decision_pending": True,
    }


def _paths(cfg) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m4_negative_saturation_{cfg.dataset}"
    return {
        "summary_csv": root / f"{stem}_summary.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_negative_saturation_diagnostic(cfg=None) -> dict[str, str]:
    cfg = cfg or configure_negative_saturation_diagnostic()
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = relation._prepare_and_load(cfg)
    model.eval()
    membership, thresholds = relation._membership(prepared, axes)
    users = np.asarray(prepared["cache"].users, dtype=np.int64)
    user_embedding, item_embedding = model.propagate_pref()

    rng = np.random.default_rng(SAMPLING_SEED)
    pair_users, pair_positives = sample_user_positives(
        users,
        prepared["data"]["csr_ptr"],
        prepared["data"]["csr_items"],
        rng,
    )
    negatives = m4_helpers.sample_uniform_negative_matrix(
        pair_users,
        pair_positives,
        int(prepared["data"]["n_items"]),
        prepared["data"]["pos_key"],
        rng,
        k=CANDIDATES,
    )
    signals = pair_signals(
        user_embedding, item_embedding, pair_users, pair_positives, negatives
    )
    del user_embedding, item_embedding

    per_user = per_user_table(pair_users, signals, membership)
    summary = summarize(per_user)
    bootstrap = bootstrap_high_minus_low(per_user)
    reading = dataset_reading(summary, bootstrap)

    paths = _paths(cfg)
    test10._atomic_csv(paths["summary_csv"], summary)
    test10._atomic_csv(paths["per_user_csv"], per_user)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "record": record,
            },
            "group_thresholds": thresholds,
            "sampled_pairs": int(len(pair_users)),
            "summary_rows": summary.to_dict("records"),
            "bootstrap": bootstrap,
            "reading": reading,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    print("\n1) CLV 구간별 학습신호")
    print(summary.to_string(index=False))
    print("\n2) 고CLV − 저CLV 95% bootstrap 구간")
    print(json.dumps(bootstrap["contrasts"], ensure_ascii=False, indent=2))
    print("\n3) 이 데이터의 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n결과 파일:", {key: str(value) for key, value in paths.items()})
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_negative_saturation_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
