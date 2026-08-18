"""Fast Dunnhumby screen for axis-specific non-negative CLV gates.

The existing jointly trained ID|N|V LightGCN is unchanged.  Only the user
allocation rule changes: N activity ranks scale the N block, while V
transaction-value ranks scale the V block.  Both gates stay positive and are
normalized to mean one over users with a valid axis representation.
"""

from __future__ import annotations

import json

import pandas as pd

import lightgcn_clv_equal_gate as equal
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-axis-specific-nonnegative-gate-v1"
MODEL_ID = "m2_axis_specific_gate"


def configure_axis_specific_gate_dunnhumby_run(
    **overrides,
) -> joint.JointNVConfig:
    """Dunnhumby seed-42 validation-only M1 versus axis-specific M2."""
    defaults = {
        "gate_shape": "axis_positive",
        "gamma_init": 0.1,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_axis_specific_nonnegative_gate_v1"
        ),
    }
    cfg = joint.configure_joint_nv_run(
        "dunnhumby", short_hm=False, **(defaults | overrides)
    )
    return validate_axis_specific_gate_config(cfg)


def validate_axis_specific_gate_config(
    cfg: joint.JointNVConfig,
) -> joint.JointNVConfig:
    joint.validate_joint_nv_config(cfg)
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "window_days": None,
        "input_days": 365,
        "gate_shape": "axis_positive",
        "id_dim": 64,
        "axis_dim": 16,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "eval_test": False,
        "eval_holdout": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                "axis-specific gate screening requires "
                f"{key}={expected!r}"
            )
    return cfg


def preflight_summary(cfg: joint.JointNVConfig) -> dict:
    cfg = validate_axis_specific_gate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": ["m1", MODEL_ID],
        "architecture": (
            "existing ID|N|V layer-0 blocks -> one binary LightGCN -> "
            "one final dot score"
        ),
        "axis_specific_allocation": {
            "N": (
                "g_N(u)=(0.5+q_N(u))/mean_valid(0.5+q_N); "
                "applied only to the user N block"
            ),
            "V": (
                "g_V(u)=(0.5+q_V(u))/mean_valid(0.5+q_V); "
                "applied only to the user V block"
            ),
            "range": "strictly positive; no centered sign reversal",
            "mean_strength": "one separately on each valid axis",
        },
        "clv_interpretation": (
            "historical CLV is retained as its N and V components; "
            "no common total-CLV scalar gate is used"
        ),
        "preference_preservation": (
            "BPR(S_ID) + BPR(stopgrad(S_ID)+S_N+S_V)"
        ),
        "gamma": (
            "learned sqrt-gamma on both user/item sides; initial gamma=0.1"
        ),
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "m4_sample_weighting": False,
        "selection_rules": {
            equal.SELECTION_PRIMARY: "maximum validation recall@10",
            equal.SELECTION_ECONOMIC: (
                "maximum price/purchase-amount weighted hit@10 among epochs "
                "passing the same six 99% M1 accuracy guardrails"
            ),
        },
        "selection_symmetry": (
            "the identical rules and guard thresholds are applied to M1 and M2"
        ),
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "out_dir": cfg.out_dir,
    }


def run_experiment(
    cfg: joint.JointNVConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_axis_specific_gate_config(
        cfg or configure_axis_specific_gate_dunnhumby_run()
    )
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = joint._prepare(cfg)
    canonical_metrics, canonical_training = equal._canonical_m1_reference(
        prepared, cfg
    )
    guard_thresholds = {
        metric: 0.99 * float(canonical_metrics[metric])
        for metric in equal.GUARD_METRICS
    }
    guard_reference = {
        "source": "established same-seed M1 validation checkpoint",
        "metrics": {
            metric: float(canonical_metrics[metric])
            for metric in equal.GUARD_METRICS
        },
        "thresholds_99pct": guard_thresholds,
        "training": canonical_training,
    }
    print("대칭 경제 체크포인트의 정확도 하한:")
    print(json.dumps(guard_thresholds, ensure_ascii=False, indent=2))

    fresh_m1 = equal._fresh_m1(prepared, cfg)
    runs = {
        "m1": equal._train_with_symmetric_selection(
            fresh_m1,
            list(fresh_m1.pref_params()),
            prepared,
            cfg,
            "m1_axis_specific_gate",
            guard_thresholds,
        )
    }
    v3.set_seed(cfg.seed)
    m2 = joint._build_model(prepared, cfg, "joint_nv")
    runs[MODEL_ID] = equal._train_with_symmetric_selection(
        m2,
        list(m2.parameters()),
        prepared,
        cfg,
        MODEL_ID,
        guard_thresholds,
    )
    rows = equal._result_rows(
        cfg,
        runs,
        model_id=MODEL_ID,
        gate_label="axis_positive",
    )
    decision = equal.screening_decision(rows, model_id=MODEL_ID)
    paired = equal._paired_rows(runs, model_id=MODEL_ID)
    frame = equal._persist(
        prepared,
        cfg,
        rows,
        paired,
        runs,
        guard_reference,
        decision,
        stem_prefix="m2_axis_specific_gate_dunnhumby",
        code_version=CODE_VERSION,
        preflight=preflight,
    )
    print("축별 비음수 게이트 M2 판정:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_axis_specific_gate_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
