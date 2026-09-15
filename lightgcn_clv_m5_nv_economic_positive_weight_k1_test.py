"""Dunnhumby ten-seed rerun with one-negative BPR in M1/M2/M4/M5."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m5_economic_positive_weight_test as base
import lightgcn_clv_m5_nv_economic_positive_weight_k1 as screen
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-explicit-nv-personalized-positive-weight-single-negative-bpr-test10-v1"
M5NVEconomicPositiveK1TestConfig = base.M5EconomicPositiveTestConfig
FULL_SEEDS = base.FULL_SEEDS


def configure_m5_nv_economic_positive_k1_test_run(
    **overrides,
) -> M5NVEconomicPositiveK1TestConfig:
    defaults = {
        "dataset": "dunnhumby",
        "seeds": FULL_SEEDS,
        "negative_count": 1,
        "reused_seed42_json": "",
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_explicit_nv_personalized_positive_weight_"
            "single_negative_bpr_test10_v1"
        ),
    }
    return validate_k1_test_config(
        M5NVEconomicPositiveK1TestConfig(**(defaults | overrides))
    )


def validate_k1_test_config(
    cfg: M5NVEconomicPositiveK1TestConfig,
) -> M5NVEconomicPositiveK1TestConfig:
    required = {
        "dataset": "dunnhumby",
        "seeds": FULL_SEEDS,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 4,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.15,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 1,
        "batch_size": 8192,
        "lr": 5e-4,
        "pref_reg": 1e-3,
        "input_days": 365,
        "shuffle_degree_bins": 10,
        "reused_seed42_json": "",
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"단일 음성 BPR 10시드 재실험은 {key}={expected!r}이어야 합니다"
            )
    if not cfg.out_dir:
        raise ValueError("단일 음성 BPR 10시드 재실험에는 out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5NVEconomicPositiveK1TestConfig) -> dict:
    cfg = validate_k1_test_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "models": list(screen.MODEL_IDS),
        "research_question": (
            "When every arm uses the LightGCN original-style one-positive/"
            "one-uniform-negative BPR comparison, how do the prior M2, M4, "
            "and M5 interventions compare with M1?"
        ),
        "protocol_status": (
            "post-hoc baseline-loss correction on an already exposed test; "
            "not a fresh confirmatory test"
        ),
        "single_changed_factor_from_prior_ten_seed_run": {
            "negative_count": "5 -> 1",
            "bpr": "mean over five negatives -> one positive/negative pair",
        },
        "unchanged_from_prior_ten_seed_run": {
            "training": "DAY 1--697 (former train + validation)",
            "test": "DAY 698--704",
            "validation_constructed": False,
            "early_stopping": False,
            "holdout_constructed": False,
            "epochs": cfg.epochs,
            "id_dim": cfg.id_dim,
            "n_layers": cfg.n_layers,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "pref_reg": cfg.pref_reg,
            "graph": "binary",
            "negative_sampling": "uniform",
            "min_item_interactions": 1,
            "rho": cfg.rho,
            "positive_weight_lambda": cfg.positive_weight_lambda,
        },
        "baseline": {
            "model": "LightGCN",
            "score": "dot product of propagated user/item ID embeddings",
            "loss": "softplus(s(u,j)-s(u,i+)) plus sampled layer-0 L2",
            "negative_sampling": "one uniformly sampled unobserved item per positive",
            "clv_or_economic_input": False,
            "sample_weight": False,
        },
        "m2": {
            "same_as_prior_ten_seed_run": True,
            "architecture": "ID64 plus jointly propagated explicit N/V economic block",
            "q_n": "post-projection strength gate",
            "v": "q_V plus shrunken four-bin spending profile determines direction",
            "rho": cfg.rho,
        },
        "m4": {
            "same_as_prior_ten_seed_run": True,
            "positive_row_weight": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit, "
                "normalized over all training positive rows"
            ),
            "q_c": "percentile of historical n_u*v_u",
        },
        "m5": "the unchanged M2 representation and M4 row weight in one training loop",
        "task_guards": {
            "new_item_task": True,
            "train_pairs_excluded_from_test_truth_and_candidates": True,
            "min_item_interactions": 1,
            "one_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "reporting": {
            "primary_output": "ten-seed means and same-seed paired differences",
            "all_accuracy_economic_exposure_segment_metrics": True,
            "model_selection_on_this_test": False,
            "significance_claim": False,
        },
        "out_dir": cfg.out_dir,
    }


def _prepare(cfg: M5NVEconomicPositiveK1TestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = base._base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    base.validate_final_test_data(data, cfg.dataset)
    if data.get("loss_w") is not None:
        raise RuntimeError("M5 자체 구현 외의 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
    economic = screen.build_nv_economic_inputs(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=q_n,
        q_v=q_v,
        q_c=q_c,
        clv_valid=clv_valid,
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
        degree_bins=cfg.shuffle_degree_bins,
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        axes["clv_proxy"],
        thresholds,
        data["n_items"],
    )
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "config_hash": base._config_hash(cfg, input_hash, revision),
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
        **economic,
    }


def _prepare_seed_assignments(prepared: dict, seed: int, degree_bins: int) -> None:
    """No shuffled assignments are needed in the requested four-arm rerun."""

    del prepared, seed, degree_bins


def _load_reused_seed42_arms(
    prepared: dict,
    cfg: M5NVEconomicPositiveK1TestConfig,
) -> list[dict]:
    """K=5 results cannot be reused after changing the BPR sampling protocol."""

    del prepared, cfg
    return []


def run_m5_nv_economic_positive_k1_test(
    cfg: M5NVEconomicPositiveK1TestConfig | None = None,
) -> pd.DataFrame:
    """Train and evaluate all four arms for seeds 42--51."""

    cfg = cfg or configure_m5_nv_economic_positive_k1_test_run()
    cfg = configure_m5_nv_economic_positive_k1_test_run(**cfg.__dict__)
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, "CODE_VERSION", CODE_VERSION))
        stack.enter_context(patch.object(base, "MODEL_IDS", screen.MODEL_IDS))
        stack.enter_context(patch.object(base, "screen", screen))
        stack.enter_context(patch.object(base, "validate_test_config", validate_k1_test_config))
        stack.enter_context(patch.object(base, "preflight_summary", preflight_summary))
        stack.enter_context(patch.object(base, "_prepare", _prepare))
        stack.enter_context(
            patch.object(base, "_prepare_seed_assignments", _prepare_seed_assignments)
        )
        stack.enter_context(
            patch.object(base, "_load_reused_seed42_arms", _load_reused_seed42_arms)
        )
        result = base.run_m5_economic_positive_test(cfg)
    result.attrs["preflight"] = preflight_summary(cfg)
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m5_nv_economic_positive_k1_test_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
