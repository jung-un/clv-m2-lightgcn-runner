"""Seed-42 M1--M5 test runner using only one user historical-CLV scalar."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from clv_m3_direct_value_graph import build_direct_clv_value_graph
import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m5_economic_positive_weight_test as base
import lightgcn_clv_minimal_scalar_factorial as screen
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-minimal-scalar-historical-clv-m1-m5-test-seed42-v1"
MinimalScalarCLVTestConfig = base.M5EconomicPositiveTestConfig
PILOT_SEEDS = base.PILOT_SEEDS


def configure_minimal_scalar_clv_test_run(**overrides) -> MinimalScalarCLVTestConfig:
    defaults = {
        "dataset": "dunnhumby",
        "seeds": PILOT_SEEDS,
        "economic_dim": 1,
        "negative_count": 1,
        "rho": 0.15,
        "positive_weight_lambda": 0.5,
        "reused_seed42_json": "",
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_minimal_scalar_historical_clv_m1_m5_test_seed42_v1"
        ),
    }
    return validate_minimal_scalar_clv_test_config(
        MinimalScalarCLVTestConfig(**(defaults | overrides))
    )


def validate_minimal_scalar_clv_test_config(
    cfg: MinimalScalarCLVTestConfig,
) -> MinimalScalarCLVTestConfig:
    required = {
        "dataset": "dunnhumby",
        "seeds": PILOT_SEEDS,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 1,
        "rho": 0.15,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 1,
        "batch_size": 8192,
        "lr": 5e-4,
        "pref_reg": 1e-3,
        "input_days": 365,
        "reused_seed42_json": "",
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"최소 scalar CLV seed-42 실험은 {key}={expected!r}이어야 합니다"
            )
    if not cfg.out_dir:
        raise ValueError("최소 scalar CLV 실험에는 out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: MinimalScalarCLVTestConfig) -> dict:
    cfg = validate_minimal_scalar_clv_test_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": 42,
        "models": list(screen.MODEL_IDS),
        "research_question": (
            "What happens when the same user historical CLV proxy is inserted "
            "directly at the M2 expression, M3 graph, and M4 loss locations?"
        ),
        "historical_clv_proxy": {
            "raw": "C_u = n_u * v_u from merged training history",
            "model_input": "q_C(u) = mid-rank percentile of C_u",
            "future_clv_claim": False,
        },
        "m1": "binary ID-only LightGCN and one uniform negative BPR",
        "m2": (
            "append centered q_C(u) to the user and mean centered purchaser q_C "
            "to the item at layer 0; jointly propagate a learned 1x1 projection"
        ),
        "m3": (
            "same observed edge set; edge weight mean-one(1 + q_C(u)); "
            "symmetric LightGCN normalization"
        ),
        "m4": (
            "positive-row weight (1 + 0.5*q_C(u)) divided by its mean over "
            "all training positive rows"
        ),
        "m5": "M2 expression + M3 graph + M4 loss in one training loop",
        "excluded": [
            "separate q_N/q_V embeddings",
            "price or price-bin inputs",
            "category inputs",
            "personalized economic fit",
            "external reranking",
            "shuffle or degree controls in this first directional check",
        ],
        "fixed": {
            "training": "DAY 1--697",
            "test": "DAY 698--704",
            "validation_constructed": False,
            "holdout_constructed": False,
            "new_item_task": True,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "negative_count": cfg.negative_count,
            "graph": "binary except the declared M3 arms",
            "negative_sampling": "uniform",
            "one_optimizer_per_arm": True,
        },
        "prior_m3_c3_disclosure": (
            "direct user-CLV edge weighting was previously weak; it is retained "
            "only as the simplest M3 reference and a component of the new minimal "
            "joint comparison"
        ),
        "interpretation": (
            "single-seed post-hoc directional result on an exposed test; report "
            "all favorable and unfavorable metrics and do not tune on this result"
        ),
        "out_dir": cfg.out_dir,
    }


def _prepare(cfg: MinimalScalarCLVTestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = base._base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    base.validate_final_test_data(data, cfg.dataset)
    if data.get("loss_w") is not None:
        raise RuntimeError("최소 scalar CLV 모형 외의 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    _q_n, _q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
    scalar = screen.build_minimal_scalar_clv_inputs(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_c=q_c,
        clv_valid=clv_valid,
    )

    direct_graph = build_direct_clv_value_graph(
        data["train"],
        data["n_users"],
        data["n_items"],
        q_c,
        alpha=screen.M3_ALPHA,
    )
    expected_keys = np.unique(
        data["train"]["u_idx"].to_numpy(np.int64) * data["n_items"]
        + data["train"]["i_idx"].to_numpy(np.int64)
    )
    if not (
        np.array_equal(direct_graph.edge_users, expected_keys // data["n_items"])
        and np.array_equal(direct_graph.edge_items, expected_keys % data["n_items"])
    ):
        raise RuntimeError("M3의 엣지집합이 M1 binary 그래프와 다릅니다")
    clv_adj = v3.build_adj(
        direct_graph.edge_users,
        direct_graph.edge_items,
        direct_graph.user_clv_weights,
        data["n_users"],
        data["n_items"],
    )
    scalar["economic_input_diagnostics"]["m3_edge_weight_diagnostics"] = (
        direct_graph.diagnostics["user_clv_weights"]
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
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
        "clv_adj": clv_adj,
        **scalar,
    }


def _prepare_seed_assignments(prepared: dict, seed: int, degree_bins: int) -> None:
    del prepared, seed, degree_bins


def _load_reused_seed42_arms(
    prepared: dict, cfg: MinimalScalarCLVTestConfig
) -> list[dict]:
    del prepared, cfg
    return []


def run_minimal_scalar_clv_test(
    cfg: MinimalScalarCLVTestConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or configure_minimal_scalar_clv_test_run()
    cfg = configure_minimal_scalar_clv_test_run(**cfg.__dict__)
    with ExitStack() as stack:
        stack.enter_context(patch.object(base, "CODE_VERSION", CODE_VERSION))
        stack.enter_context(patch.object(base, "MODEL_IDS", screen.MODEL_IDS))
        stack.enter_context(patch.object(base, "screen", screen))
        stack.enter_context(
            patch.object(
                base,
                "validate_test_config",
                validate_minimal_scalar_clv_test_config,
            )
        )
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
            preflight_summary(configure_minimal_scalar_clv_test_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
