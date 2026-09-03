"""Ten-seed attribution screen for the rho=.05 CLV embedding candidate."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_constrained_economic_embedding as single
import lightgcn_clv_gradient_isolated_economic_interaction as helpers
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-level-composition-price-embedding-multiseed-v1"
SEEDS = tuple(range(42, 52))
MATCHED_MODEL_ID = single.MATCHED_MODEL_ID
MODEL_ID = single.MODEL_ID
SHUFFLED_MODEL_ID = single.SHUFFLED_MODEL_ID
ID_ONLY_MODEL_ID = single.ID_ONLY_MODEL_ID
MODELS = (
    MATCHED_MODEL_ID,
    MODEL_ID,
    SHUFFLED_MODEL_ID,
    ID_ONLY_MODEL_ID,
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)
PRIMARY_METRICS = (
    "고CLV_recall@10",
    "고CLV_ndcg@10",
    "price_purchase_amount_weighted_hit@10",
)
MIN_POSITIVE_SEED_COUNT = 7


@dataclass(frozen=True)
class M2MultiSeedConfig:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = SEEDS
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 3
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
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


def configure_multiseed_run(**overrides) -> M2MultiSeedConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_level_composition_price_embedding_multiseed_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
        "seed42_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_level_composition_price_embedding_historical_screen_v1"
        ),
    }
    return validate_multiseed_config(
        M2MultiSeedConfig(**(defaults | overrides))
    )


def validate_multiseed_config(cfg: M2MultiSeedConfig) -> M2MultiSeedConfig:
    required = {
        "dataset": "dunnhumby",
        "seeds": SEEDS,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 3,
        "rho": 0.05,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "input_days": 365,
        "shuffle_degree_bins": 10,
        "shuffle_seed_offset": 1000,
        "minimum_positive_seed_count": MIN_POSITIVE_SEED_COUNT,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M2 10-seed 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("M2 10-seed 학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.seed42_result_dir:
        raise ValueError("M2 10-seed 결과·기준 경로가 모두 필요합니다")
    return cfg


def preflight_summary(cfg: M2MultiSeedConfig) -> dict:
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
        "m2": {
            "architecture": "ID(64)|CLV relation(2)|explicit price fit(1)",
            "rho": cfg.rho,
            "item_price_budget": cfg.item_price_budget,
            "structure_changed_from_seed42": False,
            "degree_matched_clv_shuffle": True,
            "id_only_is_posthoc_view_without_extra_training": True,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "paired_controls": {
            "matched_rho0": MATCHED_MODEL_ID,
            "degree_matched_clv_assignment": SHUFFLED_MODEL_ID,
            "jointly_trained_id_only": ID_ONLY_MODEL_ID,
            "same_initialization_batches_and_negatives": True,
        },
        "decision": {
            "overall_guard": (
                "the 10-seed mean of every Recall/NDCG@10/20/50 must not be "
                "below matched rho=0"
            ),
            "direct_and_assignment_attribution": (
                "actual CLV must have a positive mean delta and win at least "
                "7/10 seeds for all primary metrics against matched rho=0, "
                "joint ID-only, and degree-matched shuffle"
            ),
            "primary_metrics": list(PRIMARY_METRICS),
        },
        "statistical_note": (
            "10 development seeds summarize training randomness; no population "
            "significance or final-test claim"
        ),
        "automatic_epoch_resume": True,
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: M2MultiSeedConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _single_config(cfg: M2MultiSeedConfig, *, seed: int) -> single.ConstrainedEconomicConfig:
    return single.ConstrainedEconomicConfig(
        dataset=cfg.dataset,
        seed=seed,
        time_cutoff=cfg.time_cutoff,
        evaluation_days=cfg.evaluation_days,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=cfg.rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=50,
        include_degree_matched_shuffle=False,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
        shuffle_seed=cfg.shuffle_seed_offset + seed,
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )


def _prepare(cfg: M2MultiSeedConfig) -> dict:
    prepared = single._prepare(_single_config(cfg, seed=42))
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["out_dir"] = Path(cfg.out_dir)
    return prepared


def degree_matched_clv_shuffle(
    prepared: dict, cfg: M2MultiSeedConfig, *, seed: int
) -> dict:
    """Jointly permute N, V, and CLV inside user-degree deciles."""

    return single._degree_matched_clv_shuffle(
        prepared, _single_config(cfg, seed=seed)
    )


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{seed}"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(
    prepared: dict, *, model_id: str, seed: int, rho: float, assignment: str
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "seed": seed,
        "rho": rho,
        "assignment": assignment,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _load_checkpoint(path: Path, prepared: dict, model, *, seed: int, rho: float):
    checkpoint = helpers._load_state(path)
    if checkpoint.get("input_hash") != prepared["input_hash"]:
        raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
    if int(checkpoint.get("config", {}).get("seed", -1)) != seed:
        raise RuntimeError("cached checkpoint의 seed가 다릅니다")
    if float(checkpoint.get("rho", -1.0)) != rho:
        raise RuntimeError("cached checkpoint의 rho가 다릅니다")
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()


def _legacy_seed42_arm(
    prepared: dict,
    cfg: M2MultiSeedConfig,
    *,
    model_id: str,
    rho: float,
    model,
) -> dict | None:
    if model_id not in {MATCHED_MODEL_ID, MODEL_ID}:
        return None
    root = Path(cfg.seed42_result_dir)
    candidates = sorted(root.glob(f"arms/*/{model_id}_s42.json"))
    valid = []
    for result_path in candidates:
        checkpoint_path = result_path.with_suffix(".pt")
        if not checkpoint_path.exists():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            _load_checkpoint(
                checkpoint_path, prepared, model, seed=42, rho=rho
            )
        except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        valid.append((result_path, checkpoint_path, payload))
    if not valid:
        return None
    metric_snapshots = {_canonical(entry[2].get("metrics", {})) for entry in valid}
    if len(metric_snapshots) != 1:
        raise RuntimeError(f"seed 42 {model_id} 기존 결과가 서로 일치하지 않습니다")
    result_path, checkpoint_path, payload = valid[-1]
    _load_checkpoint(checkpoint_path, prepared, model, seed=42, rho=rho)
    reused = dict(payload)
    reused["role"] = "reused_matched_control" if rho == 0 else "reused_model"
    reused["source_result"] = str(result_path)
    print(f"  [reused] 기존 seed 42 {model_id} checkpoint 재사용")
    return reused


def _run_arm(
    prepared: dict,
    cfg: M2MultiSeedConfig,
    *,
    seed: int,
    model_id: str,
    rho: float,
    assignment: dict | None = None,
    assignment_name: str = "observed",
):
    arm_cfg = _single_config(cfg, seed=seed)
    paths = _arm_paths(prepared, model_id, seed)
    model, params = single._build_model(
        prepared, arm_cfg, rho, clv_assignment=assignment
    )
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        _load_checkpoint(paths["checkpoint"], prepared, model, seed=seed, rho=rho)
        print(f"  [cached] seed {seed} | {model_id}")
        return payload, model
    if seed == 42 and assignment_name == "observed":
        reused = _legacy_seed42_arm(
            prepared, cfg, model_id=model_id, rho=rho, model=model
        )
        if reused is not None:
            return reused, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_multiseed_train",
            model_id=model_id,
            seed=seed,
            config_hash=_arm_hash(
                prepared,
                model_id=model_id,
                seed=seed,
                rho=rho,
                assignment=assignment_name,
            ),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = test10._fixed_epoch_train(
        model, params, prepared, arm_cfg, model_id, seed, store
    )
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "seed": seed,
            "rho": rho,
            "clv_assignment": assignment_name,
            "config": asdict(arm_cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": model_id,
        "role": {
            MATCHED_MODEL_ID: "matched_control",
            MODEL_ID: "model",
            SHUFFLED_MODEL_ID: "assignment_control",
        }[model_id],
        "seed": seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": rho,
        "clv_assignment": assignment_name,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": model.representation_diagnostics(),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload, model


def _id_only_payload(active_model, prepared: dict, cfg: M2MultiSeedConfig, seed: int):
    view = shared._IDOnlyView(active_model).to(v3.DEVICE)
    metrics, _ = moe._flat_evaluation(
        view,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    return {
        "model_id": ID_ONLY_MODEL_ID,
        "role": "joint_training_ablation",
        "seed": seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": {},
        "training": {"additional_training": False},
    }


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        rows.append(
            {
                "seed": arm["seed"],
                "model_id": arm["model_id"],
                "role": arm["role"],
                "split": arm.get("split", "historical_development_days_684_690"),
                "final_epoch": arm["final_epoch"],
                **arm.get("diagnostics", {}),
                **arm["metrics"],
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "model_id"]).reset_index(drop=True)


def multiseed_decision(absolute: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    expected = {(seed, model) for seed in SEEDS for model in MODELS}
    observed = set(zip(absolute["seed"], absolute["model_id"], strict=False))
    if observed != expected or len(absolute) != len(expected):
        raise ValueError("10개 seed와 네 비교 view가 각각 정확히 한 행이어야 합니다")
    indexed = absolute.set_index(["seed", "model_id"])
    paired_rows = []
    for reference in (MATCHED_MODEL_ID, ID_ONLY_MODEL_ID, SHUFFLED_MODEL_ID):
        for metric in PRIMARY_METRICS:
            deltas = np.asarray(
                [
                    indexed.loc[(seed, MODEL_ID), metric]
                    - indexed.loc[(seed, reference), metric]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            positive_count = int((deltas > 0).sum())
            paired_rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "mean_delta": float(deltas.mean()),
                    "sd_delta": float(deltas.std(ddof=1)),
                    "positive_seed_count": positive_count,
                    "passes": bool(
                        deltas.mean() > 0
                        and positive_count >= MIN_POSITIVE_SEED_COUNT
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    accuracy_ratios = {}
    for metric in ACCURACY_METRICS:
        actual_mean = float(
            absolute.loc[absolute.model_id == MODEL_ID, metric].mean()
        )
        matched_mean = float(
            absolute.loc[absolute.model_id == MATCHED_MODEL_ID, metric].mean()
        )
        accuracy_ratios[metric] = actual_mean / matched_mean
    overall_guard = all(ratio >= 1.0 for ratio in accuracy_ratios.values())
    return {
        "positive_screen": bool(overall_guard and paired["passes"].all()),
        "all_overall_metrics_not_below_matched": bool(overall_guard),
        "all_primary_control_comparisons_pass": bool(paired["passes"].all()),
        "accuracy_mean_ratios_vs_matched_rho0": accuracy_ratios,
        "minimum_positive_seed_count": MIN_POSITIVE_SEED_COUNT,
        "statistical_note": (
            "10 development seeds; mean, SD, and paired win count describe "
            "training randomness without a population significance claim"
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


def run_multiseed(cfg: M2MultiSeedConfig | None = None) -> pd.DataFrame:
    cfg = validate_multiseed_config(cfg or configure_multiseed_run())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    shuffle_diagnostics = {}
    for seed in cfg.seeds:
        print(f"\n===== M2 rho=.05 multiseed | seed {seed} =====")
        shuffle = degree_matched_clv_shuffle(prepared, cfg, seed=seed)
        shuffle_diagnostics[str(seed)] = {
            key: value
            for key, value in shuffle.items()
            if key not in {"q_n", "q_v", "q_c", "clv_valid", "source_user", "stratum", "user_degree"}
        }
        matched, _ = _run_arm(
            prepared,
            cfg,
            seed=seed,
            model_id=MATCHED_MODEL_ID,
            rho=0.0,
        )
        active, active_model = _run_arm(
            prepared,
            cfg,
            seed=seed,
            model_id=MODEL_ID,
            rho=cfg.rho,
        )
        shuffled, _ = _run_arm(
            prepared,
            cfg,
            seed=seed,
            model_id=SHUFFLED_MODEL_ID,
            rho=cfg.rho,
            assignment=shuffle,
            assignment_name="degree_matched_shuffle",
        )
        arms.extend(
            [
                matched,
                active,
                shuffled,
                _id_only_payload(active_model, prepared, cfg, seed),
            ]
        )

    absolute = _absolute_rows(arms)
    decision, paired = multiseed_decision(absolute)
    metric_summary = _metric_summary(absolute)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"m2_clv_level_composition_price_multiseed_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "metric_summary_csv": out / f"{stem}_mean.csv",
        "paired_decision_csv": out / f"{stem}_paired.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], absolute)
    test10._atomic_csv(paths["metric_summary_csv"], metric_summary)
    test10._atomic_csv(paths["paired_decision_csv"], paired)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": absolute.to_dict("records"),
            "metric_summary_rows": metric_summary.to_dict("records"),
            "paired_control_rows": paired.to_dict("records"),
            "decision": decision,
            "shuffle_diagnostics": shuffle_diagnostics,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    absolute.attrs["metric_summary"] = metric_summary
    absolute.attrs["paired_control"] = paired
    absolute.attrs["decision"] = decision
    absolute.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    key_metrics = set(ACCURACY_METRICS + PRIMARY_METRICS)
    print("\n10시드 절대지표:")
    print(absolute.to_string(index=False))
    print("\n10시드 평균·표준편차:")
    print(metric_summary[metric_summary.metric.isin(key_metrics)].to_string(index=False))
    print("\n동일 seed 대응 비교:")
    print(paired.to_string(index=False))
    print("\n사전 판정:", decision)
    print("결과 파일:", absolute.attrs["result_paths"])
    return absolute


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_multiseed_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
