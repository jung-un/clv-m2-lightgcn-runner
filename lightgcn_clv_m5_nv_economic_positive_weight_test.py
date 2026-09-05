"""Test-only seed-42 runner for the explicit q_N/q_V M2 plus M4 M5."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m5_economic_positive_weight_test as base
import lightgcn_clv_m5_nv_economic_positive_weight as screen
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-explicit-nv-economic-positive-weighting-test-only-v1"
M5NVEconomicPositiveTestConfig = base.M5EconomicPositiveTestConfig
PILOT_SEEDS = base.PILOT_SEEDS
_LEGACY_PREFLIGHT = base.preflight_summary


def configure_m5_nv_economic_positive_test_run(
    **overrides,
) -> M5NVEconomicPositiveTestConfig:
    defaults = {
        "dataset": "dunnhumby",
        "seeds": PILOT_SEEDS,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_explicit_nv_economic_positive_weighting_test_seed42_v1"
        ),
    }
    cfg = M5NVEconomicPositiveTestConfig(**(defaults | overrides))
    cfg = base.validate_test_config(cfg)
    if cfg.dataset != "dunnhumby" or cfg.seeds != PILOT_SEEDS:
        raise ValueError("이 새 N/V 모형은 Dunnhumby seed 42 기술 실험만 허용합니다")
    if cfg.reused_seed42_json:
        raise ValueError("새 N/V 모형은 이전 seed 42 결과를 재사용할 수 없습니다")
    return cfg


def preflight_summary(cfg: M5NVEconomicPositiveTestConfig) -> dict:
    cfg = configure_m5_nv_economic_positive_test_run(**cfg.__dict__)
    summary = _LEGACY_PREFLIGHT(cfg)
    summary.update(
        {
            "code_version": CODE_VERSION,
            "models": list(screen.MODEL_IDS),
            "research_axis": "M5 partial combination: explicit-N/V M2 plus M4 loss",
            "protocol_status": (
                "exploratory only because this Dunnhumby test interval was already exposed"
            ),
        }
    )
    summary["m2"] = {
        "architecture": "ID64 plus one jointly propagated 4D N/V-economic block",
        "q_n": "post-projection strength gate only",
        "v": "q_V plus shrunken four-bin spending profile determines direction",
        "item": "overall amount percentile plus centered four-bin basis",
        "q_c_used_in_m2": False,
        "category_relative_amount_used": False,
        "economic_bins": cfg.economic_bins,
        "shrinkage_strength": cfg.shrinkage_strength,
        "rho": cfg.rho,
        "joint_layer0_propagation": True,
    }
    summary["m4_prime"].update(
        {
            "q_c_role": "positive-row learning priority only",
            "formula": "1 + lambda*q_C*(2*item_amount_percentile-1)",
        }
    )
    summary["decision"] = {
        "primary": "M5 vNDCG@10 must exceed M1, joint CLV shuffle, and degree gate",
        "interaction_required": False,
        "accuracy_guard_required": False,
        "accuracy_reporting": "all six Recall/NDCG metrics are still reported",
        "exposure_guard": "coverage/distinct >=95% and top10 share <=105% of M1",
        "test_selection": "the result must not be used to retune this exposed test",
    }
    return summary


def _prepare(cfg: M5NVEconomicPositiveTestConfig) -> dict:
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
    prepared["joint_shuffle"] = screen.joint_degree_matched_shuffle(
        prepared, seed=seed, degree_bins=degree_bins
    )
    prepared["degree_gate"] = {
        "q_n": prepared["q_n"],
        "q_v": prepared["q_v"],
        "q_c": prepared["degree_percentile"],
        "clv_valid": prepared["clv_valid"],
        "user_activity_gate": prepared["user_activity_gate"],
        "user_economic_input": prepared["user_economic_input"],
        "user_economic_valid": prepared["user_economic_valid"],
    }


def run_m5_nv_economic_positive_test(
    cfg: M5NVEconomicPositiveTestConfig | None = None,
) -> pd.DataFrame:
    """Run through the tested generic harness without mutating the legacy files."""

    cfg = cfg or configure_m5_nv_economic_positive_test_run()
    cfg = configure_m5_nv_economic_positive_test_run(**cfg.__dict__)
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, "CODE_VERSION", CODE_VERSION))
        stack.enter_context(patch.object(base, "MODEL_IDS", screen.MODEL_IDS))
        stack.enter_context(patch.object(base, "screen", screen))
        stack.enter_context(patch.object(base, "preflight_summary", preflight_summary))
        stack.enter_context(patch.object(base, "_prepare", _prepare))
        stack.enter_context(
            patch.object(base, "_prepare_seed_assignments", _prepare_seed_assignments)
        )
        result = base.run_m5_economic_positive_test(cfg)
    result.attrs["preflight"] = preflight_summary(cfg)
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m5_nv_economic_positive_test_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
