"""Ten-seed historical screen for CLV-conditioned hard-negative BPR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_m4_clv_hard_negative as single
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-clv-conditioned-hard-negative-multiseed-v1"
SEEDS = tuple(range(42, 52))
K1_MODEL_ID = single.K1_MODEL_ID
MEAN_K5_MODEL_ID = single.MEAN_K5_MODEL_ID
M4_MODEL_ID = single.M4_MODEL_ID
SHUFFLED_M4_MODEL_ID = "m4_clv_hard_k5_degree_shuffled"
MODELS = (
    K1_MODEL_ID,
    MEAN_K5_MODEL_ID,
    M4_MODEL_ID,
    SHUFFLED_M4_MODEL_ID,
)
PRIMARY_METRICS = (
    "고CLV_recall@10",
    "고CLV_ndcg@10",
    "price_purchase_amount_weighted_hit@10",
)
ACCURACY_METRICS = single.ACCURACY_METRICS
MIN_POSITIVE_SEED_COUNT = 7


@dataclass(frozen=True)
class M4MultiSeedConfig:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = SEEDS
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    shuffle_degree_bins: int = 10
    shuffle_seed_offset: int = 1000
    minimum_positive_seed_count: int = MIN_POSITIVE_SEED_COUNT
    out_dir: str = ""
    baseline_result_dir: str = ""
    seed42_result_dir: str = ""


def configure_multiseed_run(**overrides) -> M4MultiSeedConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m4_clv_hard_negative_multiseed_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
        "seed42_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m4_clv_hard_negative_historical_screen_v1"
        ),
    }
    return validate_multiseed_config(
        M4MultiSeedConfig(**(defaults | overrides))
    )


def validate_multiseed_config(cfg: M4MultiSeedConfig) -> M4MultiSeedConfig:
    required = {
        "dataset": "dunnhumby",
        "seeds": SEEDS,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "shuffle_degree_bins": 10,
        "shuffle_seed_offset": 1000,
        "minimum_positive_seed_count": MIN_POSITIVE_SEED_COUNT,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M4 10-seed 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("M4 10-seed 학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.seed42_result_dir:
        raise ValueError("M4 10-seed 결과·기준 경로가 모두 필요합니다")
    return cfg


def preflight_summary(cfg: M4MultiSeedConfig) -> dict:
    cfg = validate_multiseed_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "models": list(MODELS),
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "id_dim": cfg.id_dim,
            "layers": cfg.n_layers,
            "negative_count": cfg.negative_count,
            "epochs": cfg.epochs,
            "epoch_selection": False,
            "min_item_interactions": 1,
        },
        "paired_controls": {
            "multi_negative": MEAN_K5_MODEL_ID,
            "degree_matched_clv_assignment": SHUFFLED_M4_MODEL_ID,
            "same_initialization_batches_and_negatives": True,
        },
        "primary_metrics": list(PRIMARY_METRICS),
        "minimum_positive_seed_count": cfg.minimum_positive_seed_count,
        "decision": (
            "actual CLV must have positive 10-seed mean delta and win at least "
            "7/10 seeds for every primary metric against both controls"
        ),
        "statistical_note": (
            "historical development seeds measure training randomness; "
            "no population significance claim"
        ),
        "automatic_epoch_resume": True,
        "out_dir": cfg.out_dir,
    }


def degree_matched_q_clv_shuffle(
    q_clv: np.ndarray,
    valid: np.ndarray,
    user_degree: np.ndarray,
    *,
    n_bins: int,
    seed: int,
) -> dict:
    """Permute q_CLV only inside binary user-degree rank strata."""

    q_clv = np.asarray(q_clv, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    user_degree = np.asarray(user_degree, dtype=np.int64)
    if q_clv.shape != valid.shape or q_clv.shape != user_degree.shape:
        raise ValueError("q_CLV·유효성·user degree shape이 다릅니다")
    if n_bins <= 0:
        raise ValueError("degree 구간 수는 양수여야 합니다")
    valid_index = np.flatnonzero(valid & (user_degree > 0))
    if len(valid_index) < 2:
        raise RuntimeError("degree-matched CLV 순열의 유효 고객이 부족합니다")

    ranks = pd.Series(user_degree[valid_index]).rank(method="average").to_numpy()
    strata_valid = np.floor(
        (ranks - 0.5) * n_bins / len(valid_index)
    ).astype(np.int16)
    strata_valid = np.minimum(strata_valid, n_bins - 1)
    strata = np.full(len(q_clv), -1, dtype=np.int16)
    strata[valid_index] = strata_valid
    source = np.arange(len(q_clv), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for stratum in np.unique(strata_valid):
        target = valid_index[strata_valid == stratum]
        if len(target) < 2:
            continue
        permuted = rng.permutation(target)
        if np.array_equal(permuted, target):
            permuted = np.roll(permuted, 1)
        source[target] = permuted

    changed = valid_index[source[valid_index] != valid_index]
    if not len(changed):
        raise RuntimeError("degree-matched CLV 순열이 고객 배정을 바꾸지 못했습니다")
    if np.any(strata[changed] != strata[source[changed]]):
        raise RuntimeError("CLV 순열이 binary user-degree 구간을 벗어났습니다")
    shuffled = q_clv[source]
    shuffled[~valid] = 0.0
    return {
        "q_clv": shuffled,
        "source_user": source,
        "stratum": strata,
        "changed_valid_user_share": float(len(changed) / len(valid_index)),
        "unique_user_q_mean_actual": float(q_clv[valid_index].mean()),
        "unique_user_q_mean_shuffled": float(shuffled[valid_index].mean()),
        "training_row_q_mean_actual": float(
            np.average(q_clv[valid_index], weights=user_degree[valid_index])
        ),
        "training_row_q_mean_shuffled": float(
            np.average(shuffled[valid_index], weights=user_degree[valid_index])
        ),
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: M4MultiSeedConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _single_config(
    cfg: M4MultiSeedConfig,
    *,
    seed: int,
    negative_count: int,
    out_dir: str,
) -> single.M4HardNegativeConfig:
    return single.M4HardNegativeConfig(
        dataset=cfg.dataset,
        seed=seed,
        time_cutoff=cfg.time_cutoff,
        evaluation_days=cfg.evaluation_days,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        negative_count=negative_count,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        out_dir=out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )


def _prepare(cfg: M4MultiSeedConfig) -> dict:
    base = _single_config(
        cfg, seed=42, negative_count=cfg.negative_count, out_dir=cfg.out_dir
    )
    prepared = single._prepare(base)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["out_dir"] = Path(cfg.out_dir)
    train_edges = prepared["data"]["train"][["u_idx", "i_idx"]].drop_duplicates()
    prepared["binary_user_degree"] = np.bincount(
        train_edges["u_idx"].to_numpy(np.int64),
        minlength=prepared["data"]["n_users"],
    )
    return prepared


def _assignment_diagnostics(prepared: dict, q_clv: np.ndarray) -> dict:
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    degree = np.asarray(prepared["binary_user_degree"], dtype=np.int64)
    valid_index = np.flatnonzero(valid & (degree > 0))
    train_users = prepared["data"]["tr_u"]
    row_q = np.asarray(q_clv, dtype=np.float64)[train_users]
    return {
        "unique_user_q_mean": float(np.asarray(q_clv)[valid_index].mean()),
        "training_row_q_mean": float(row_q.mean()),
        "expected_hardest_weight_mean_k5": float(0.2 + 0.8 * row_q.mean()),
    }


def _load_seed42_cached_arms(cfg: M4MultiSeedConfig, manifest: list[dict]):
    root = Path(cfg.seed42_result_dir)
    found = []
    for path in sorted(root.glob("m4_clv_hard_negative_*.json")):
        if path.name.endswith("_comparison.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload["config"]
            if (
                payload.get("code_version") == single.CODE_VERSION
                and config.get("seed") == 42
                and config.get("negative_count") == 5
                and config.get("epochs") == 100
                and _canonical(payload.get("input_manifest")) == _canonical(manifest)
                and set(payload.get("arms", {}))
                >= {MEAN_K5_MODEL_ID, M4_MODEL_ID}
            ):
                found.append(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not found:
        return None
    snapshots = {
        _canonical(
            {
                model_id: payload["arms"][model_id]["metrics"]
                for model_id in (MEAN_K5_MODEL_ID, M4_MODEL_ID)
            }
        )
        for payload in found
    }
    if len(snapshots) != 1:
        raise RuntimeError("seed 42 기존 M4 결과가 서로 일치하지 않습니다")
    return found[-1]["arms"]


def _baseline_payload(prepared: dict) -> dict:
    row = prepared["baseline"]
    metrics = {
        key: value
        for key, value in row.items()
        if isinstance(value, (int, float, np.number))
        and ("@" in key or key == "user_value_tendency_recommended_price_alignment")
    }
    return {
        "seed": 42,
        "model_id": K1_MODEL_ID,
        "role": "baseline",
        "final_epoch": 100,
        "negative_count": 1,
        "metrics": metrics,
        "training": {"reused_seed42_baseline": True},
    }


def _run_arm(
    prepared: dict,
    cfg: M4MultiSeedConfig,
    *,
    seed: int,
    model_id: str,
    q_clv: np.ndarray,
) -> dict:
    negative_count = 1 if model_id == K1_MODEL_ID else cfg.negative_count
    arm_cfg = _single_config(
        cfg,
        seed=seed,
        negative_count=negative_count,
        out_dir=cfg.out_dir,
    )
    arm_prepared = dict(prepared)
    arm_prepared["q_c"] = np.asarray(q_clv, dtype=np.float32)
    payload = dict(single._run_arm(arm_prepared, arm_cfg, model_id))
    payload["role"] = {
        K1_MODEL_ID: "baseline",
        MEAN_K5_MODEL_ID: "multineg_control",
        M4_MODEL_ID: "model",
        SHUFFLED_M4_MODEL_ID: "assignment_control",
    }[model_id]
    payload["assignment_diagnostics"] = _assignment_diagnostics(
        prepared, arm_prepared["q_c"]
    )
    return payload


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        rows.append(
            {
                "seed": arm["seed"],
                "model_id": arm["model_id"],
                "role": arm["role"],
                "final_epoch": arm["final_epoch"],
                "negative_count": arm["negative_count"],
                **arm.get("assignment_diagnostics", {}),
                **arm.get("training", {}).get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "model_id"]).reset_index(
        drop=True
    )


def multiseed_decision(absolute: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    expected = {(seed, model) for seed in SEEDS for model in MODELS}
    observed = set(zip(absolute["seed"], absolute["model_id"], strict=False))
    if observed != expected or len(absolute) != len(expected):
        raise ValueError("10개 seed와 네 arm이 각각 정확히 한 행이어야 합니다")
    indexed = absolute.set_index(["seed", "model_id"])
    paired_rows = []
    for reference in (MEAN_K5_MODEL_ID, SHUFFLED_M4_MODEL_ID):
        for metric in PRIMARY_METRICS:
            deltas = np.array(
                [
                    indexed.loc[(seed, M4_MODEL_ID), metric]
                    - indexed.loc[(seed, reference), metric]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            paired_rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "mean_delta": float(deltas.mean()),
                    "sd_delta": float(deltas.std(ddof=1)),
                    "positive_seed_count": int((deltas > 0).sum()),
                    "passes": bool(
                        deltas.mean() > 0
                        and (deltas > 0).sum() >= MIN_POSITIVE_SEED_COUNT
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    accuracy_ratios = {}
    for metric in ACCURACY_METRICS:
        actual_mean = float(
            absolute.loc[absolute.model_id == M4_MODEL_ID, metric].mean()
        )
        baseline_mean = float(
            absolute.loc[absolute.model_id == K1_MODEL_ID, metric].mean()
        )
        accuracy_ratios[metric] = actual_mean / baseline_mean
    weighted_actual = float(
        absolute.loc[
            absolute.model_id == M4_MODEL_ID,
            "price_purchase_amount_weighted_hit@10",
        ].mean()
    )
    weighted_m1 = float(
        absolute.loc[
            absolute.model_id == K1_MODEL_ID,
            "price_purchase_amount_weighted_hit@10",
        ].mean()
    )
    m1_guard = all(value >= 0.99 for value in accuracy_ratios.values())
    m1_economic = weighted_actual > weighted_m1
    return {
        "positive_screen": bool(paired["passes"].all() and m1_guard and m1_economic),
        "all_control_comparisons_pass": bool(paired["passes"].all()),
        "m1_accuracy_guard_pass": bool(m1_guard),
        "m1_economic_guard_pass": bool(m1_economic),
        "accuracy_mean_ratios_vs_m1": accuracy_ratios,
        "minimum_positive_seed_count": MIN_POSITIVE_SEED_COUNT,
        "statistical_note": (
            "10 development seeds; report mean, SD, and win count without "
            "population significance claim"
        ),
    }, paired


def _metric_summary(absolute: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in absolute.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]
    rows = []
    for model_id, group in absolute.groupby("model_id", sort=False):
        for metric in metric_columns:
            values = group[metric].to_numpy(np.float64)
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


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_multiseed(cfg: M4MultiSeedConfig | None = None) -> pd.DataFrame:
    cfg = validate_multiseed_config(cfg or configure_multiseed_run())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    cached_seed42 = _load_seed42_cached_arms(cfg, prepared["manifest"])
    zeros = np.zeros_like(prepared["q_c"], dtype=np.float32)
    arms = []
    shuffle_diagnostics = {}
    for seed in cfg.seeds:
        print(f"\n===== M4 hard-negative multiseed | seed {seed} =====")
        shuffle = degree_matched_q_clv_shuffle(
            prepared["q_c"],
            prepared["clv_valid"],
            prepared["binary_user_degree"],
            n_bins=cfg.shuffle_degree_bins,
            seed=cfg.shuffle_seed_offset + seed,
        )
        shuffle_diagnostics[str(seed)] = {
            key: value
            for key, value in shuffle.items()
            if key not in {"q_clv", "source_user", "stratum"}
        }
        if seed == 42:
            arms.append(_baseline_payload(prepared))
        else:
            arms.append(
                _run_arm(
                    prepared,
                    cfg,
                    seed=seed,
                    model_id=K1_MODEL_ID,
                    q_clv=zeros,
                )
            )
        for model_id, q_values in (
            (MEAN_K5_MODEL_ID, zeros),
            (M4_MODEL_ID, prepared["q_c"]),
        ):
            if seed == 42 and cached_seed42 is not None:
                cached = dict(cached_seed42[model_id])
                cached["role"] = (
                    "multineg_control" if model_id == MEAN_K5_MODEL_ID else "model"
                )
                cached["assignment_diagnostics"] = _assignment_diagnostics(
                    prepared, q_values
                )
                arms.append(cached)
            else:
                arms.append(
                    _run_arm(
                        prepared,
                        cfg,
                        seed=seed,
                        model_id=model_id,
                        q_clv=q_values,
                    )
                )
        arms.append(
            _run_arm(
                prepared,
                cfg,
                seed=seed,
                model_id=SHUFFLED_M4_MODEL_ID,
                q_clv=shuffle["q_clv"],
            )
        )

    absolute = _absolute_rows(arms)
    decision, paired = multiseed_decision(absolute)
    metric_summary = _metric_summary(absolute)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"m4_clv_hard_negative_multiseed_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "metric_summary_csv": out / f"{stem}_mean.csv",
        "paired_decision_csv": out / f"{stem}_paired.csv",
        "json": out / f"{stem}.json",
    }
    absolute.to_csv(paths["absolute_csv"], index=False)
    metric_summary.to_csv(paths["metric_summary_csv"], index=False)
    paired.to_csv(paths["paired_decision_csv"], index=False)
    _atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": absolute.to_dict("records"),
            "metric_summary_rows": metric_summary.to_dict("records"),
            "paired_control_rows": paired.to_dict("records"),
            "decision": decision,
            "shuffle_diagnostics": shuffle_diagnostics,
            "seed42_reused": cached_seed42 is not None,
        },
    )
    absolute.attrs["metric_summary"] = metric_summary.to_dict("records")
    absolute.attrs["paired_control"] = paired.to_dict("records")
    absolute.attrs["decision"] = decision
    absolute.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    return absolute
