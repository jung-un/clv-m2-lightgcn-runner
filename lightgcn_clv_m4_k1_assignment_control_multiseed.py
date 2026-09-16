"""Ten-seed development replication of the K=1 M4 assignment control.

The M4 formula and all training behavior come from
``lightgcn_clv_m4_k1_assignment_control_screen``.  This module only repeats
the frozen three-arm comparison over seeds 42--51, resumes completed arms,
and summarizes paired seed differences.  It never constructs the protected
final test or a holdout split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m4_k1_assignment_control_screen as single
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_k1_m4_improvement_screen as training
import lightgcn_clv_m5_clv_scaled_value_basis_k1_screen as screen
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-personalized-positive-weight-k1-assignment-control-multiseed-v1"
FULL_SEEDS = tuple(range(42, 52))
MIN_POSITIVE_SEED_COUNT = 7
MODEL_IDS = single.MODEL_IDS
M1_MODEL_ID = single.M1_MODEL_ID
M4_ACTUAL_MODEL_ID = single.M4_ACTUAL_MODEL_ID
M4_SHUFFLED_MODEL_ID = single.M4_SHUFFLED_MODEL_ID
ECONOMIC_METRICS = single.ECONOMIC_METRICS
ACCURACY_METRICS = single.ACCURACY_METRICS
ACCURACY_GUARD = single.ACCURACY_GUARD


@dataclass(frozen=True)
class M4K1AssignmentMultiSeedConfig(training.M5K1ImprovementConfig):
    seeds: tuple[int, ...] = FULL_SEEDS
    minimum_positive_seed_count: int = MIN_POSITIVE_SEED_COUNT
    seed42_result_dir: str = ""


def configure_m4_k1_assignment_control_multiseed(
    **overrides,
) -> M4K1AssignmentMultiSeedConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": (
            f"{data_root}_m4_k1_assignment_control_development_multiseed_v1"
        ),
        "baseline_result_dir": (
            f"{data_root}_m2_repeatshare_historical_backtest_v1"
        ),
        "seed42_result_dir": (
            f"{data_root}_m4_k1_assignment_control_development_screen_v1"
        ),
    }
    return validate_config(M4K1AssignmentMultiSeedConfig(**(defaults | overrides)))


def validate_config(
    cfg: M4K1AssignmentMultiSeedConfig,
) -> M4K1AssignmentMultiSeedConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "seeds": FULL_SEEDS,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 3,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.25,
        "gate_delta": 0.25,
        "basis_bandwidth": 0.25,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 1,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "minimum_positive_seed_count": MIN_POSITIVE_SEED_COUNT,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M4 K=1 10-seed 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("M4 K=1 10-seed 학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.seed42_result_dir:
        raise ValueError("결과·기준·seed 42 재사용 경로가 모두 필요합니다")
    return cfg


def preflight_summary(cfg: M4K1AssignmentMultiSeedConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "split": "historical_development_days_684_690",
        "trained_models_per_seed": list(MODEL_IDS),
        "research_question": (
            "Across ten training seeds, does the K=1 personalized positive-row "
            "M4 improve new-item recommendation, and is the gain consistently "
            "larger with the observed historical q_C assignment than with a "
            "degree-matched q_C permutation?"
        ),
        "m4": {
            "historical_clv_proxy": "q_C=percentile(n_u*v_u)",
            "raw_positive_row_weight": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "normalization": "divide by the mean raw weight over all train rows",
            "positive_weight_lambda": cfg.positive_weight_lambda,
        },
        "control": {
            "permuted": "q_C only",
            "stratum": (
                f"{cfg.shuffle_degree_bins} binary user-degree rank strata, "
                "CLV-valid users only"
            ),
            "shuffle_seed": "the training seed; seed 42 exactly matches the pilot",
            "held_fixed": [
                "item amount percentile",
                "observed user economic-bin fit",
                "binary graph",
                "single uniform negative stream",
                "BPR loss, initialization and optimizer",
            ],
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_truth_and_candidates": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_count": 1,
            "negative_sampling": "one uniformly sampled unseen item",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "external_reranking": False,
        },
        "reading_rule": {
            "primary_metrics": list(ECONOMIC_METRICS),
            "mean_direction": (
                "actual M4 has a positive 10-seed mean paired delta against "
                "both M1 and the degree-matched q_C shuffle on both primary metrics"
            ),
            "paired_seed_consistency": (
                f"actual M4 wins at least {cfg.minimum_positive_seed_count}/10 "
                "seeds against each reference on both primary metrics"
            ),
            "accuracy_guard": (
                "the 10-seed mean of each Recall/NDCG@10,@20,@50 for actual "
                "M4 is at least 99% of the corresponding M1 mean"
            ),
            "multiseed_assignment_pass": (
                "all mean-direction, paired-seed-consistency, and accuracy "
                "conditions hold"
            ),
        },
        "seed42_handling": (
            "reuse only a complete, input- and config-matched three-arm pilot; "
            "otherwise retrain all three seed-42 arms"
        ),
        "automatic_epoch_resume": True,
        "statistical_note": (
            "ten exposed development seeds summarize training randomness; "
            "paired t intervals are descriptive and no population significance, "
            "generalization, or final-test claim is made"
        ),
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: M4K1AssignmentMultiSeedConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODEL_IDS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _single_config(
    cfg: M4K1AssignmentMultiSeedConfig, *, seed: int
) -> training.M5K1ImprovementConfig:
    names = {field.name for field in fields(training.M5K1ImprovementConfig)}
    values = {name: getattr(cfg, name) for name in names}
    values.update(seed=int(seed), shuffle_seed=int(seed), out_dir=cfg.out_dir)
    return training.M5K1ImprovementConfig(**values)


def _prepare(cfg: M4K1AssignmentMultiSeedConfig) -> dict:
    prepared = training._prepare(_single_config(cfg, seed=42))
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["out_dir"] = Path(cfg.out_dir)
    return prepared


def _seed_prepared(
    prepared: dict,
    cfg: M4K1AssignmentMultiSeedConfig,
    *,
    seed: int,
) -> tuple[dict, training.M5K1ImprovementConfig, dict]:
    arm_cfg = _single_config(cfg, seed=seed)
    current = dict(prepared)
    shuffle = single.degree_matched_q_c_shuffle(current, arm_cfg)
    current["q_c_shuffle"] = shuffle
    return current, arm_cfg, shuffle


_PATH_FIELDS = {
    "out_dir",
    "baseline_result_dir",
    "m1_reference_json",
    "m4_reference_json",
}


def _checkpoint_matches(checkpoint: dict, prepared: dict, arm_cfg) -> bool:
    if checkpoint.get("input_hash") != prepared["input_hash"]:
        return False
    stored = checkpoint.get("config", {})
    for field in fields(training.M5K1ImprovementConfig):
        if field.name in _PATH_FIELDS:
            continue
        if stored.get(field.name) != getattr(arm_cfg, field.name):
            return False
    return True


def _load_seed42_pilot(
    prepared: dict,
    cfg: M4K1AssignmentMultiSeedConfig,
    seed_prepared: dict,
    arm_cfg: training.M5K1ImprovementConfig,
) -> tuple[dict[str, dict], dict[str, object]] | None:
    """Reuse the complete pilot only when all three arms match exactly."""

    root = Path(cfg.seed42_result_dir)
    candidates = sorted(root.glob("m4_k1_assignment_control_*.json"))
    valid_runs = []
    for result_path in candidates:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("code_version") != single.CODE_VERSION:
            continue
        stored_cfg = payload.get("config", {})
        if any(
            stored_cfg.get(field.name) != getattr(arm_cfg, field.name)
            for field in fields(training.M5K1ImprovementConfig)
            if field.name not in _PATH_FIELDS
        ):
            continue
        stored_arms = payload.get("arms", {})
        if set(stored_arms) != set(MODEL_IDS):
            continue

        loaded_arms: dict[str, dict] = {}
        loaded_models: dict[str, object] = {}
        failed = False
        for spec in single.arm_specifications(seed_prepared):
            model_id = spec["model_id"]
            arm = stored_arms[model_id]
            checkpoint_path = Path(arm.get("checkpoint", ""))
            if not checkpoint_path.exists():
                failed = True
                break
            checkpoint = legacy.load_checkpoint_or_discard(checkpoint_path)
            if checkpoint is None or not _checkpoint_matches(
                checkpoint, prepared, arm_cfg
            ):
                failed = True
                break
            model = screen._build_model(
                seed_prepared,
                arm_cfg,
                {"m2_assignment": seed_prepared["m2_actual"], **spec},
            )
            model.load_state_dict(checkpoint["state"], strict=True)
            model.to(v3.DEVICE)
            model.eval()
            loaded_arms[model_id] = dict(arm)
            loaded_models[model_id] = model
        if not failed:
            valid_runs.append((result_path, loaded_arms, loaded_models))

    if not valid_runs:
        return None
    snapshots = {
        _canonical(
            {
                model_id: run[1][model_id].get("metrics", {})
                for model_id in MODEL_IDS
            }
        )
        for run in valid_runs
    }
    if len(snapshots) != 1:
        raise RuntimeError("서로 다른 seed 42 M4 pilot 결과가 여러 개 발견됐습니다")
    result_path, arms, models = valid_runs[-1]
    print(f"  [reused] 완료된 seed 42 세 arm 재사용: {result_path}")
    return arms, models


def _run_seed(
    prepared: dict,
    cfg: M4K1AssignmentMultiSeedConfig,
    *,
    seed: int,
) -> tuple[list[dict], dict, dict, list[dict]]:
    seed_prepared, arm_cfg, shuffle = _seed_prepared(
        prepared, cfg, seed=seed
    )
    reused = (
        _load_seed42_pilot(prepared, cfg, seed_prepared, arm_cfg)
        if seed == 42
        else None
    )
    if reused is None:
        arms: dict[str, dict] = {}
        models: dict[str, object] = {}
        for spec in single.arm_specifications(seed_prepared):
            print(
                f"\n===== seed {seed} | {spec['model_id']} | K=1 | "
                f"fixed {arm_cfg.epochs} epochs ====="
            )
            arm, model = training._run_arm(
                single._arm_prepared(seed_prepared, spec), arm_cfg, spec
            )
            arm["m4_assignment"] = spec["m4_assignment"]
            arms[spec["model_id"]] = arm
            models[spec["model_id"]] = model
    else:
        arms, models = reused

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], seed_prepared, max_k=arm_cfg.diagnostic_max_k
        )
        for model_id in MODEL_IDS
    }
    users = topk[M1_MODEL_ID][0]
    if not all(np.array_equal(users, topk[mid][0]) for mid in MODEL_IDS):
        raise RuntimeError("arm별 평가 사용자 순서가 다릅니다")
    overlaps = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[reference][1],
                topk[model_id][1],
                seed_prepared["cache"].seg,
            )
            .assign(seed=seed, reference=reference, model_id=model_id)
            for reference, model_id in (
                (M1_MODEL_ID, M4_ACTUAL_MODEL_ID),
                (M1_MODEL_ID, M4_SHUFFLED_MODEL_ID),
                (M4_SHUFFLED_MODEL_ID, M4_ACTUAL_MODEL_ID),
            )
        ],
        ignore_index=True,
    )
    overall = overlaps[overlaps.group.eq("전체")].set_index(
        ["reference", "model_id"]
    )
    change_share = float(
        overall.at[
            (M4_SHUFFLED_MODEL_ID, M4_ACTUAL_MODEL_ID),
            "top10_set_changed_user_share",
        ]
    )
    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in MODEL_IDS}
    reading = single.attribution_reading(
        metric_rows,
        actual_vs_shuffle_top10_change_share=change_share,
        row_weight_cvs={
            model_id: float(
                arms[model_id]["weight_diagnostics"]["row_weight_cv"]
            )
            for model_id in (M4_ACTUAL_MODEL_ID, M4_SHUFFLED_MODEL_ID)
        },
    )
    rows = [
        {
            "seed": seed,
            "model_id": model_id,
            "role": arms[model_id]["role"],
            "split": arms[model_id]["split"],
            "final_epoch": arms[model_id]["final_epoch"],
            "m4_assignment": arms[model_id].get(
                "m4_assignment",
                {
                    M1_MODEL_ID: "inactive",
                    M4_ACTUAL_MODEL_ID: "observed_q_c",
                    M4_SHUFFLED_MODEL_ID: "degree_matched_q_c_shuffle",
                }[model_id],
            ),
            "positive_weight_lambda": (
                cfg.positive_weight_lambda if model_id != M1_MODEL_ID else 0.0
            ),
            **arms[model_id]["weight_diagnostics"],
            **arms[model_id]["metrics"],
        }
        for model_id in MODEL_IDS
    ]
    shuffle_diagnostics = {
        key: value
        for key, value in shuffle.items()
        if key not in {"q_clv", "q_c", "source_user", "stratum"}
    }
    shuffle_diagnostics.update(
        q_c_multiset_preserved=shuffle["q_c_multiset_preserved"],
        invalid_q_c_zero_preserved=shuffle["invalid_q_c_zero_preserved"],
        degree_stratum_preserved=shuffle["degree_stratum_preserved"],
    )
    del models, topk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, reading, shuffle_diagnostics, overlaps.to_dict("records")


def _metric_columns(absolute: pd.DataFrame) -> list[str]:
    return [
        column
        for column in absolute.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]


def seedwise_comparison(absolute: pd.DataFrame) -> pd.DataFrame:
    indexed = absolute.set_index(["seed", "model_id"])
    rows = []
    for seed in FULL_SEEDS:
        actual = indexed.loc[(seed, M4_ACTUAL_MODEL_ID)]
        for reference in (M1_MODEL_ID, M4_SHUFFLED_MODEL_ID):
            baseline = indexed.loc[(seed, reference)]
            for metric in _metric_columns(absolute):
                reference_value = float(baseline[metric])
                model_value = float(actual[metric])
                rows.append(
                    {
                        "seed": seed,
                        "reference": reference,
                        "model_id": M4_ACTUAL_MODEL_ID,
                        "metric": metric,
                        "reference_value": reference_value,
                        "model_value": model_value,
                        "absolute_delta": model_value - reference_value,
                        "relative_change_pct": (
                            100.0 * (model_value - reference_value) / reference_value
                            if reference_value != 0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (reference, metric), group in comparison.groupby(
        ["reference", "metric"], sort=False
    ):
        values = group.sort_values("seed")["absolute_delta"].to_numpy(np.float64)
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        half = (
            float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
            if n > 1
            else np.nan
        )
        rows.append(
            {
                "reference": reference,
                "metric": metric,
                "n_seeds": n,
                "mean_delta": mean,
                "sd_delta": sd,
                "ci95_low": mean - half if n > 1 else np.nan,
                "ci95_high": mean + half if n > 1 else np.nan,
                "positive_seed_count": int((values > 0).sum()),
                "nonnegative_seed_count": int((values >= 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(absolute: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, group in absolute.groupby("model_id", sort=False):
        for metric in _metric_columns(absolute):
            values = group.sort_values("seed")[metric].to_numpy(np.float64)
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def multiseed_decision(
    absolute: pd.DataFrame, paired: pd.DataFrame
) -> dict:
    expected = {(seed, model_id) for seed in FULL_SEEDS for model_id in MODEL_IDS}
    observed = set(zip(absolute.seed, absolute.model_id, strict=False))
    if observed != expected or len(absolute) != len(expected):
        raise ValueError("seeds 42~51의 세 arm이 각각 정확히 한 행이어야 합니다")

    primary = paired[
        paired.metric.isin(ECONOMIC_METRICS)
        & paired.reference.isin((M1_MODEL_ID, M4_SHUFFLED_MODEL_ID))
    ].copy()
    primary["passes"] = (
        (primary.mean_delta > 0)
        & (primary.positive_seed_count >= MIN_POSITIVE_SEED_COUNT)
    )
    mean_metrics = absolute.groupby("model_id")[_metric_columns(absolute)].mean()
    accuracy_ratios = {
        metric: float(
            mean_metrics.loc[M4_ACTUAL_MODEL_ID, metric]
            / mean_metrics.loc[M1_MODEL_ID, metric]
        )
        for metric in ACCURACY_METRICS
    }
    accuracy_guard = all(value >= ACCURACY_GUARD for value in accuracy_ratios.values())
    comparison_passes = {
        reference: bool(primary[primary.reference.eq(reference)].passes.all())
        for reference in (M1_MODEL_ID, M4_SHUFFLED_MODEL_ID)
    }
    passed = bool(accuracy_guard and all(comparison_passes.values()))
    return {
        "classification": (
            "q_c_assignment_supported_across_training_seeds"
            if passed
            else "q_c_assignment_not_supported_across_training_seeds"
        ),
        "multiseed_assignment_pass": passed,
        "actual_beats_m1_on_both_primary_metrics": comparison_passes[M1_MODEL_ID],
        "actual_beats_shuffle_on_both_primary_metrics": comparison_passes[
            M4_SHUFFLED_MODEL_ID
        ],
        "six_accuracy_mean_metrics_at_least_99pct_of_m1": bool(accuracy_guard),
        "accuracy_mean_ratios_actual_vs_m1": accuracy_ratios,
        "minimum_positive_seed_count": MIN_POSITIVE_SEED_COUNT,
        "primary_paired_results": primary.to_dict("records"),
        "statistical_note": (
            "ten exposed development seeds summarize training randomness; "
            "paired t intervals are descriptive and no population significance, "
            "generalization, or final-test claim is made"
        ),
    }


def run_m4_k1_assignment_control_multiseed(
    cfg: M4K1AssignmentMultiSeedConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(
        cfg or configure_m4_k1_assignment_control_multiseed()
    )
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    absolute_rows = []
    per_seed_readings = {}
    shuffle_diagnostics = {}
    overlap_rows = []
    for seed in cfg.seeds:
        print(f"\n######## seed {seed} / M4 K=1 assignment control ########")
        rows, reading, shuffle, overlaps = _run_seed(
            prepared, cfg, seed=seed
        )
        absolute_rows.extend(rows)
        per_seed_readings[str(seed)] = reading
        shuffle_diagnostics[str(seed)] = shuffle
        overlap_rows.extend(overlaps)

    absolute = (
        pd.DataFrame(absolute_rows)
        .sort_values(["seed", "model_id"])
        .reset_index(drop=True)
    )
    comparison = seedwise_comparison(absolute)
    paired = paired_summary(comparison)
    means = metric_summary(absolute)
    decision = multiseed_decision(absolute, paired)
    overlaps = pd.DataFrame(overlap_rows)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"m4_k1_assignment_control_multiseed_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "metric_summary_csv": out / f"{stem}_mean.csv",
        "paired_summary_csv": out / f"{stem}_paired.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], absolute)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["metric_summary_csv"], means)
    legacy.test10._atomic_csv(paths["paired_summary_csv"], paired)
    legacy.test10._atomic_csv(paths["top10_overlap_csv"], overlaps)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": absolute.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "metric_summary_rows": means.to_dict("records"),
            "paired_summary_rows": paired.to_dict("records"),
            "top10_overlap_rows": overlaps.to_dict("records"),
            "per_seed_readings": per_seed_readings,
            "shuffle_diagnostics": shuffle_diagnostics,
            "decision": decision,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    absolute.attrs.update(
        comparison_records=comparison.to_dict("records"),
        metric_summary_records=means.to_dict("records"),
        paired_summary_records=paired.to_dict("records"),
        top10_overlap_records=overlaps.to_dict("records"),
        per_seed_readings=per_seed_readings,
        shuffle_diagnostics=shuffle_diagnostics,
        decision=decision,
        result_paths={key: str(value) for key, value in paths.items()},
    )
    print("\n1) 10시드 판독")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("\n결과 파일:", json.dumps(absolute.attrs["result_paths"], ensure_ascii=False, indent=2))
    return absolute


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m4_k1_assignment_control_multiseed()),
            ensure_ascii=False,
            indent=2,
        )
    )
