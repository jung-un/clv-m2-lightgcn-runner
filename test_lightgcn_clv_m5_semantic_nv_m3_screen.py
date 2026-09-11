import numpy as np

import lightgcn_clv_m5_semantic_nv_m3_screen as runner


def _metrics(value):
    return {
        "recall@10": 0.10 + value,
        "ndcg@10": 0.11 + value,
        "recall@20": 0.20 + value,
        "ndcg@20": 0.21 + value,
        "recall@50": 0.30 + value,
        "ndcg@50": 0.31 + value,
        "vndcg@10": 0.40 + value,
        "price_purchase_amount_weighted_hit@10": 0.50 + value,
    }


def test_preflight_freezes_m2_m4_and_trains_four_isolated_arms(tmp_path):
    cfg = runner.configure_semantic_nv_m3_screen(
        out_dir=str(tmp_path / "out"),
        baseline_result_dir=str(tmp_path / "base"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == list(runner.MODEL_IDS)
    assert summary["m2_frozen_definition"]["rho"] == 0.15
    assert summary["m2_frozen_definition"]["beta"] == 0.25
    assert summary["m4_frozen_definition"]["loss"].startswith("mean of K=5")
    assert summary["m3"]["target_log_coefficient_ratio_std"] == 0.075
    assert summary["fixed"]["test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False


def test_all_arms_share_observed_m2_and_m4_and_only_m3_gate_changes():
    prepared = {}
    cfg = runner.configure_semantic_nv_m3_screen()
    specs = runner.arm_specifications(prepared, cfg)

    assert [spec["m3_arm"] for spec in specs] == [
        "off",
        "actual",
        "shuffle",
        "relation_only",
    ]
    assert all(spec["assignment"] is prepared for spec in specs)
    assert all(spec["weighted"] is True for spec in specs)
    assert all(spec["rho"] == 0.15 for spec in specs)


def test_gate_preflight_requires_shared_qc_mask_and_degree_bin_shuffle():
    source = np.array([1, 0, 3, 2])
    actual = np.array([0.1, 0.2, 0.0, 0.8], dtype=np.float32)
    prepared = {
        "q_c": actual,
        "clv_valid": np.array([True, True, False, True]),
        "degree_bin": np.array([0, 0, 1, 1]),
        "m3_source_user": source,
        "m3_matched": type(
            "Matched",
            (),
            {
                "user_gates": {
                    "actual": actual.copy(),
                    "shuffle": actual[source].copy(),
                    "relation_only": np.array([1, 1, 0, 1], dtype=np.float32),
                }
            },
        )(),
    }
    result = runner._gate_preflight(prepared)

    assert result["all_checks_pass"] is True
    assert result["actual_active_user_count"] == 3
    assert result["relation_only_active_user_count"] == 3


def test_outcome_requires_actual_to_beat_all_three_arms_on_both_metrics():
    rows = {
        runner.M3_OFF_MODEL_ID: _metrics(0.00),
        runner.ACTUAL_MODEL_ID: _metrics(0.03),
        runner.SHUFFLE_MODEL_ID: _metrics(0.01),
        runner.RELATION_MODEL_ID: _metrics(0.02),
    }
    positive = runner.outcome_reading(rows, operational=True)
    rows[runner.SHUFFLE_MODEL_ID]["vndcg@10"] = rows[runner.ACTUAL_MODEL_ID][
        "vndcg@10"
    ]
    failed = runner.outcome_reading(rows, operational=True)

    assert positive["outcome"] == "conditional_complementarity"
    assert positive["actual_beats_m3_off"] is True
    assert positive["actual_beats_shuffle"] is True
    assert positive["actual_beats_relation_only"] is True
    assert failed["outcome"] == "m3_increment_without_clv_assignment_support"


def test_operational_failure_is_not_evaluable():
    rows = {
        runner.M3_OFF_MODEL_ID: _metrics(0.00),
        runner.ACTUAL_MODEL_ID: _metrics(0.03),
        runner.SHUFFLE_MODEL_ID: _metrics(0.01),
        runner.RELATION_MODEL_ID: _metrics(0.02),
    }
    reading = runner.outcome_reading(rows, operational=False)
    assert reading["outcome"] == "not_evaluable"
