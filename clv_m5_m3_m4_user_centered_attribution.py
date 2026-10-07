"""Two-seed development gate for the customer-centred M3+M4 candidate.

This runner adds only the two missing comparisons identified before any
10-seed final-week run:

* M4-C alone: binary LightGCN graph plus the customer-mass-preserving M4 loss.
* M5-C permutation: the same M3+M4 algorithm after jointly reassigning the
  observed q_N/q_V/q_C/valid tuple inside binary user-degree deciles.

The target customer's z_N/z_V edge signals and price-bin preference profile
are not shuffled.  M1, M3, and observed-assignment M5-C are reused from the
finished seed 42/44 development runs.  No validation choice, final-week test,
holdout, freezing, external score addition, or reranking is performed.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import clv_m5_m3_m4_split_nv_screen as prior
import clv_m5_m3_m4_user_centered_screen as observed
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m5-m3-m4-user-centered-attribution-dev-v1"
SEEDS = (42, 44)
FIXED_EPOCH = 300
DIAGNOSTIC_EPOCH = 100
DEGREE_BINS = 10
SHUFFLE_SEED_BASE = 20261007
M4_C_MODEL_ID = "m4_binary_graph_user_centered_bpr_k1"
M5_C_SHUFFLED_MODEL_ID = "m5_m3nv_graph_user_centered_m4_tuple_shuffle_bpr_k1"
ACCURACY = observed.ACCURACY
ECONOMIC = observed.ECONOMIC


def configure(seed: int, **overrides):
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_clv_m5_m3_m4_user_centered_attribution_v1",
    }
    return observed.configure(seed=seed, **(defaults | overrides))


def degree_deciles(prepared: dict, bins: int = DEGREE_BINS) -> np.ndarray:
    """Stable equal-count strata from binary user-item degree."""
    data = prepared["data"]
    users = np.asarray(data["signals_user"] if "signals_user" in data
                       else prepared["signals"]["edge_users"], dtype=np.int64)
    degree = np.bincount(users, minlength=data["n_users"])
    order = np.argsort(np.argsort(degree, kind="stable"), kind="stable")
    strata = (order * bins) // max(len(order), 1)
    return np.clip(strata, 0, bins - 1).astype(np.int64)


def degree_matched_tuple_shuffle(
    prepared: dict,
    *,
    seed: int,
    bins: int = DEGREE_BINS,
) -> tuple[dict, dict]:
    """Jointly permute q_N/q_V/q_C/valid, preserving target-side histories."""
    strata = degree_deciles(prepared, bins)
    n_users = prepared["data"]["n_users"]
    source = np.arange(n_users, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for group in range(bins):
        members = np.flatnonzero(strata == group)
        if len(members) > 1:
            permuted = rng.permutation(members)
            if np.array_equal(permuted, members):
                permuted = np.roll(permuted, 1)
            source[members] = permuted

    assignment = {
        name: np.asarray(prepared[name])[source].copy()
        for name in ("q_n", "q_v", "q_c", "clv_valid")
    }
    checks = {
        "degree_bins": bins,
        "shuffle_seed": seed,
        "moved_user_share": float(np.mean(source != np.arange(n_users))),
        "same_degree_bin": bool(np.all(strata[source] == strata)),
        "q_n_multiset_preserved": bool(np.array_equal(
            np.sort(assignment["q_n"]), np.sort(np.asarray(prepared["q_n"]))
        )),
        "q_v_multiset_preserved": bool(np.array_equal(
            np.sort(assignment["q_v"]), np.sort(np.asarray(prepared["q_v"]))
        )),
        "q_c_multiset_preserved": bool(np.array_equal(
            np.sort(assignment["q_c"]), np.sort(np.asarray(prepared["q_c"]))
        )),
        "valid_multiset_preserved": bool(np.array_equal(
            np.sort(assignment["clv_valid"]),
            np.sort(np.asarray(prepared["clv_valid"])),
        )),
        "target_z_n_z_v_unchanged": True,
        "target_user_bin_fit_unchanged": True,
        "source_user_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
    }
    required = (
        "same_degree_bin", "q_n_multiset_preserved", "q_v_multiset_preserved",
        "q_c_multiset_preserved", "valid_multiset_preserved",
    )
    if checks["moved_user_share"] <= 0 or not all(checks[key] for key in required):
        raise RuntimeError(f"CLV 공동순열 불변식 실패: {checks}")
    assignment["source_user"] = source
    assignment["degree_bin"] = strata
    return assignment, checks


def _prepared_with_assignment(prepared: dict, assignment: dict) -> dict:
    arm = dict(prepared)
    for key in ("q_n", "q_v", "q_c", "clv_valid"):
        arm[key] = np.asarray(assignment[key]).copy()
    valid = np.asarray(arm["clv_valid"], dtype=bool)
    arm["q_value"] = np.where(valid, arm["q_v"], 0.0).astype(float)
    arm["q_activity"] = np.where(valid, arm["q_n"], 0.0).astype(float)
    return arm


def _binary_graph(prepared: dict) -> dict:
    return {
        "beta": 0.0,
        "adjacency": prepared["data"]["adj"],
        "audit": {"graph": "binary", "edge_weights_changed": False},
    }


def _observed_m5_curve(cfg, input_hash: str, beta: float,
                       row_weight_sha256: str) -> tuple[pd.DataFrame, dict]:
    seed = cfg.seeds[0]
    root = Path(
        f"{v3.default_out_dir('dunnhumby')}_clv_m5_m3_m4_user_centered_s{seed}_v1"
    )
    matches = sorted(root.glob(f"{observed.CODE_VERSION}_*.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"기존 observed M5-C seed{seed} 완료 JSON이 정확히 하나 필요합니다: {root}"
        )
    path = matches[0]
    raw = path.read_bytes()
    old = json.loads(raw)
    fixed = (
        "epochs", "eval_every", "batch_size", "lr", "n_layers", "id_dim",
        "pref_reg", "negative_count", "target_cv",
    )
    if (
        old.get("seed") != seed
        or old.get("input_hash") != input_hash
        or old.get("split") != "historical_development_days_684_690"
        or old.get("final_test") is not False
        or old.get("holdout") is not False
        or any(old["config"][key] != getattr(cfg, key) for key in fixed)
        or not np.isclose(old.get("m3_beta"), beta, rtol=0, atol=1e-12)
        or old.get("m4_c_weight_audit", {}).get("sha256") != row_weight_sha256
    ):
        raise RuntimeError(f"기존 observed M5-C seed{seed}와 이번 입력·설정이 다릅니다")
    absolute = Path(old["result_paths"]["absolute_csv"])
    if not absolute.is_file():
        raise RuntimeError(f"기존 observed M5-C absolute CSV가 없습니다: {absolute}")
    curve = pd.read_csv(absolute)
    curve = curve[
        curve.model_id.eq(observed.MODEL_ID) & curve.seed.eq(seed)
    ].copy()
    expected = set(range(cfg.eval_every, cfg.epochs + 1, cfg.eval_every))
    if set(curve.epoch) != expected:
        raise RuntimeError(f"기존 observed M5-C seed{seed} 곡선이 완전하지 않습니다")
    return curve, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "absolute_path": str(absolute),
        "absolute_sha256": hashlib.sha256(absolute.read_bytes()).hexdigest(),
    }


def _arm_specs() -> tuple[dict, dict]:
    return (
        {
            "model_id": M4_C_MODEL_ID,
            "arm": "binary",
            "gamma": 0.0,
            "question": "What does M4-C add without the M3 graph?",
            "code_version": CODE_VERSION,
            "stage": "m4_user_centered_standalone_dev",
        },
        {
            "model_id": M5_C_SHUFFLED_MODEL_ID,
            "arm": "value_and_activity",
            "gamma": 1.0,
            "question": "Does observed customer-specific CLV assignment beat its degree-matched null?",
            "code_version": CODE_VERSION,
            "stage": "m5_user_centered_tuple_shuffle_dev",
        },
    )


def _metric_columns(curve: pd.DataFrame) -> list[str]:
    return [
        column for column in curve.columns
        if "@" in column or column == "user_value_tendency_recommended_price_alignment"
    ]


def comparison_table(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    pairs = (
        (m3.ARM_VALUE_ACTIVITY, m3.M1_MODEL_ID, "M3-M1"),
        (M4_C_MODEL_ID, m3.M1_MODEL_ID, "M4-C-M1"),
        (observed.MODEL_ID, m3.ARM_VALUE_ACTIVITY, "M5-C-M3"),
        (observed.MODEL_ID, M4_C_MODEL_ID, "M5-C-M4-C"),
        (observed.MODEL_ID, M5_C_SHUFFLED_MODEL_ID, "observed-shuffled"),
        (M5_C_SHUFFLED_MODEL_ID, m3.M1_MODEL_ID, "shuffled-M1"),
    )
    rows = []
    for seed in SEEDS:
        for epoch in (DIAGNOSTIC_EPOCH, FIXED_EPOCH):
            for model_id, reference, contrast in pairs:
                candidate = index.loc[(model_id, seed, epoch)]
                baseline = index.loc[(reference, seed, epoch)]
                for metric in _metric_columns(curve):
                    base = float(baseline[metric])
                    value = float(candidate[metric])
                    rows.append({
                        "seed": seed,
                        "epoch": epoch,
                        "contrast": contrast,
                        "model_id": model_id,
                        "reference": reference,
                        "metric": metric,
                        "reference_value": base,
                        "candidate_value": value,
                        "delta": value - base,
                        "ratio": value / base if base else np.nan,
                    })
    return pd.DataFrame(rows)


def factorial_table(curve: pd.DataFrame) -> pd.DataFrame:
    at = curve[curve.epoch.eq(FIXED_EPOCH)].set_index(["seed", "model_id"])
    rows = []
    for seed in SEEDS:
        m1 = at.loc[(seed, m3.M1_MODEL_ID)]
        graph = at.loc[(seed, m3.ARM_VALUE_ACTIVITY)]
        loss = at.loc[(seed, M4_C_MODEL_ID)]
        combined = at.loc[(seed, observed.MODEL_ID)]
        for metric in _metric_columns(curve):
            rows.append({
                "seed": seed,
                "epoch": FIXED_EPOCH,
                "metric": metric,
                "m3_main_effect": float(graph[metric] - m1[metric]),
                "m4_c_main_effect": float(loss[metric] - m1[metric]),
                "m5_c_minus_m3": float(combined[metric] - graph[metric]),
                "m5_c_minus_m4_c": float(combined[metric] - loss[metric]),
                "interaction": float(combined[metric] - graph[metric] - loss[metric] + m1[metric]),
            })
    return pd.DataFrame(rows)


def summary_table(curve: pd.DataFrame) -> pd.DataFrame:
    metrics = _metric_columns(curve)
    at = curve[curve.epoch.eq(FIXED_EPOCH)][["model_id", "seed", *metrics]]
    return at.groupby("model_id", sort=False)[metrics].agg(["mean", "std"]).reset_index()


def reading(curve: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    at = curve[curve.epoch.eq(FIXED_EPOCH)]
    means = at.groupby("model_id")[_metric_columns(curve)].mean()
    actual = means.loc[observed.MODEL_ID]
    baseline = means.loc[m3.M1_MODEL_ID]
    graph = means.loc[m3.ARM_VALUE_ACTIVITY]
    shuffled = means.loc[M5_C_SHUFFLED_MODEL_ID]

    accuracy_vs_m1 = all(actual[m] >= .99 * baseline[m] for m in ACCURACY)
    accuracy_vs_m3 = all(actual[m] >= .99 * graph[m] for m in ACCURACY)
    economic_vs_m1_m3 = all(actual[m] > max(baseline[m], graph[m]) for m in ECONOMIC)
    actual_beats_shuffle = all(actual[m] > shuffled[m] for m in ECONOMIC)
    attribution_rows = comparison[
        comparison.epoch.eq(FIXED_EPOCH)
        & comparison.contrast.eq("observed-shuffled")
        & comparison.metric.isin(ECONOMIC)
    ]
    positive_by_metric = {
        metric: int((attribution_rows.loc[attribution_rows.metric.eq(metric), "delta"] > 0).sum())
        for metric in ECONOMIC
    }
    return {
        "development_gate_only": True,
        "repeatedly_exposed_split": True,
        "fixed_epoch": FIXED_EPOCH,
        "seeds": list(SEEDS),
        "m5_c_accuracy_guard_99pct_vs_m1_mean": bool(accuracy_vs_m1),
        "m5_c_accuracy_guard_99pct_vs_m3_mean": bool(accuracy_vs_m3),
        "m5_c_economic_at10_above_m1_and_m3_mean": bool(economic_vs_m1_m3),
        "observed_assignment_beats_degree_matched_shuffle_on_both_economic_at10_mean": bool(actual_beats_shuffle),
        "observed_minus_shuffle_positive_seed_count": positive_by_metric,
        "ready_for_preregistered_final_10seed": bool(
            accuracy_vs_m1 and accuracy_vs_m3 and economic_vs_m1_m3 and actual_beats_shuffle
        ),
        "m4_c_standalone_is_factorial_interpretation_not_a_required_winner": True,
        "test_or_holdout_evaluated": False,
        "significance_claim": False,
        "generalization_claim": False,
        "clv_attribution_claim": False,
    }


def run(seeds: tuple[int, ...] = SEEDS, **overrides) -> dict:
    if tuple(seeds) != SEEDS:
        raise ValueError(f"사전등록된 개발 시드는 {SEEDS}만 허용합니다")
    all_curves = []
    audits = {}
    references = {}
    new_fits = []
    for seed in seeds:
        cfg = configure(seed, **overrides)
        prepared = m3._prepare(cfg)
        graph_spec = next(
            spec for spec in m3.arm_specifications()
            if spec["model_id"] == m3.ARM_VALUE_ACTIVITY
        )
        observed_graph = m3.build_arm_graph(prepared, cfg, graph_spec)
        old_curve, old_reference = prior._original_curves(cfg, observed_graph["beta"])
        actual_weights, actual_weight_audit = observed.row_weights(prepared)
        actual_curve, actual_reference = _observed_m5_curve(
            cfg,
            prepared["input_hash"],
            observed_graph["beta"],
            actual_weight_audit["sha256"],
        )

        assignment, shuffle_audit = degree_matched_tuple_shuffle(
            prepared, seed=SHUFFLE_SEED_BASE + seed
        )
        shuffled = _prepared_with_assignment(prepared, assignment)
        shuffled_graph = m3.build_arm_graph(shuffled, cfg, graph_spec)
        shuffled_weights, shuffled_weight_audit = observed.row_weights(shuffled)

        old_hash = prepared["config_hash"]
        prepared["config_hash"] = hashlib.sha256(
            (
                f"{CODE_VERSION}:{observed.LAMBDA}:{old_hash}:"
                f"{actual_weight_audit['sha256']}:{shuffle_audit['source_user_sha256']}"
            ).encode()
        ).hexdigest()[:12]
        shuffled["config_hash"] = prepared["config_hash"]
        stale = m3.clear_stale_progress(prepared)
        m4_spec, shuffled_spec = _arm_specs()
        print(json.dumps({
            "scope": "Dunnhumby historical development days 684-690",
            "seed": seed,
            "fixed_epoch": FIXED_EPOCH,
            "new_fits": [m4_spec["model_id"], shuffled_spec["model_id"]],
            "reused": [m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY, observed.MODEL_ID],
            "observed_m3_beta": observed_graph["beta"],
            "shuffled_m3_beta": shuffled_graph["beta"],
            "shuffle_audit": shuffle_audit,
        }, ensure_ascii=False, indent=2), flush=True)

        m4_arm = m3._run_arm(
            prepared,
            cfg,
            m4_spec,
            {**_binary_graph(prepared), "row_weights": actual_weights},
            seed,
        )
        shuffled_arm = m3._run_arm(
            shuffled,
            cfg,
            shuffled_spec,
            {**shuffled_graph, "row_weights": shuffled_weights},
            seed,
        )
        new_fits.extend([f"{seed}:{m4_spec['model_id']}", f"{seed}:{shuffled_spec['model_id']}"])
        all_curves.append(pd.concat([
            old_curve,
            actual_curve,
            m3.curve_table([m4_arm, shuffled_arm], {}),
        ], ignore_index=True))
        audits[str(seed)] = {
            "observed_m3": observed_graph["audit"],
            "observed_m3_beta": observed_graph["beta"],
            "m4_c_observed": actual_weight_audit,
            "shuffle": shuffle_audit,
            "shuffled_m3": shuffled_graph["audit"],
            "shuffled_m3_beta": shuffled_graph["beta"],
            "m4_c_shuffled": shuffled_weight_audit,
            "dropped_stale_progress": stale,
        }
        references[str(seed)] = {
            "m1_m3": old_reference,
            "observed_m5_c": actual_reference,
            "input_hash": prepared["input_hash"],
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
        }

    curve = pd.concat(all_curves, ignore_index=True)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("전체 비교곡선에 model/seed/epoch 중복이 있습니다")
    required_models = {
        m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY, M4_C_MODEL_ID,
        observed.MODEL_ID, M5_C_SHUFFLED_MODEL_ID,
    }
    if set(curve.model_id) != required_models:
        raise RuntimeError(f"비교 모형 집합이 다릅니다: {sorted(set(curve.model_id))}")
    comparison = comparison_table(curve)
    factorial = factorial_table(curve)
    summary = summary_table(curve)
    final_reading = reading(curve, comparison)

    out = Path(configure(SEEDS[0], **overrides).out_dir)
    run_hash = hashlib.sha256(
        f"{CODE_VERSION}:{SEEDS}:{FIXED_EPOCH}:{DEGREE_BINS}".encode()
    ).hexdigest()[:12]
    stem = f"{CODE_VERSION}_{run_hash}"
    paths = {
        "absolute_csv": out / f"{stem}_absolute.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "factorial_csv": out / f"{stem}_factorial.csv",
        "summary_csv": out / f"{stem}_summary.csv",
        "diagnostics_json": out / f"{stem}_diagnostics.json",
        "json": out / f"{stem}.json",
    }
    io._atomic_csv(paths["absolute_csv"], curve)
    io._atomic_csv(paths["comparison_csv"], comparison)
    io._atomic_csv(paths["factorial_csv"], factorial)
    io._atomic_csv(paths["summary_csv"], summary)
    io._atomic_json(paths["diagnostics_json"], audits)
    io._atomic_json(paths["json"], {
        "code_version": CODE_VERSION,
        "split": "historical_development_days_684_690",
        "final_test": False,
        "holdout": False,
        "seeds": list(SEEDS),
        "epochs": FIXED_EPOCH,
        "new_fits": new_fits,
        "new_fit_count": len(new_fits),
        "reused_models": [m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY, observed.MODEL_ID],
        "m4_c": (
            "binary graph + exp(.5*q_C(u)*(amount(i)*fit(u,i)-customer mean)); "
            "valid positive rows normalised to customer mean one"
        ),
        "permutation_control": (
            "joint q_N/q_V/q_C/valid reassignment within binary user-degree deciles; "
            "target z_N/z_V and target user-bin fit retained"
        ),
        "references": references,
        "audits": audits,
        "reading": final_reading,
        "limits": (
            "two repeatedly exposed development seeds; no significance, stability, "
            "generalization, final-test, or CLV-attribution claim"
        ),
        "result_paths": {key: str(value) for key, value in paths.items()},
    })
    print(json.dumps({
        "reading": final_reading,
        "paths": {key: str(value) for key, value in paths.items()},
    }, ensure_ascii=False, indent=2), flush=True)
    return {
        "absolute": curve,
        "comparison": comparison,
        "factorial": factorial,
        "summary": summary,
        "reading": final_reading,
        "diagnostics": audits,
        "paths": paths,
    }


if __name__ == "__main__":
    observed.self_test()
    print("self_test ok")
