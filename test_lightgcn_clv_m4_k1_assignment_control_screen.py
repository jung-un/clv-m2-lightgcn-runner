import json
from pathlib import Path

import numpy as np

import lightgcn_clv_m4_k1_assignment_control_screen as control


def _cfg(tmp_path=None):
    root = "/tmp/m4-control" if tmp_path is None else str(tmp_path / "results")
    return control.configure_m4_k1_assignment_control_screen(
        out_dir=root,
        baseline_result_dir="/tmp/base",
    )


def test_contract_is_three_fresh_k1_development_arms(tmp_path):
    cfg = _cfg(tmp_path)
    summary = control.preflight_summary(cfg)

    assert cfg.negative_count == 1
    assert cfg.seed == 42
    assert cfg.time_cutoff == 690
    assert summary["split"] == "historical_development_days_684_690"
    assert summary["trained_models"] == list(control.MODEL_IDS)
    assert summary["reused_models"] == []
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False


def test_degree_matched_shuffle_moves_only_valid_q_c_inside_strata():
    prepared = {
        "q_c": np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], dtype=np.float32),
        "clv_valid": np.array([True, True, True, True, False, False]),
        "degree": np.array([1, 2, 3, 4, 2, 4], dtype=np.float32),
    }
    cfg = control.M4K1AssignmentControlConfig(
        out_dir="/tmp/m4-control",
        baseline_result_dir="/tmp/base",
        shuffle_degree_bins=2,
        negative_count=1,
        rho=0.25,
    )

    shuffled = control.degree_matched_q_c_shuffle(prepared, cfg)

    valid = prepared["clv_valid"]
    np.testing.assert_allclose(
        np.sort(shuffled["q_c"][valid]), np.sort(prepared["q_c"][valid])
    )
    np.testing.assert_allclose(shuffled["q_c"][~valid], 0.0)
    assert shuffled["changed_valid_user_share"] > 0.0
    assert shuffled["q_c_multiset_preserved"] is True
    assert shuffled["degree_stratum_preserved"] is True


def test_arm_specs_change_only_q_c_assignment_for_the_two_m4_arms():
    prepared = {
        "q_c": np.array([0.2, 0.8], dtype=np.float32),
        "q_c_shuffle": {"q_c": np.array([0.8, 0.2], dtype=np.float32)},
    }

    specs = control.arm_specifications(prepared)

    assert [spec["model_id"] for spec in specs] == list(control.MODEL_IDS)
    assert [(spec["rho"], spec["improvement"]) for spec in specs] == [
        (0.0, None),
        (0.0, "original"),
        (0.0, "original"),
    ]
    np.testing.assert_array_equal(specs[1]["q_c"], prepared["q_c"])
    np.testing.assert_array_equal(specs[2]["q_c"], prepared["q_c_shuffle"]["q_c"])
    assert specs[1]["m4_assignment"] == "observed_q_c"
    assert specs[2]["m4_assignment"] == "degree_matched_q_c_shuffle"


def _metrics(economic, accuracy=1.0):
    return {
        "recall@10": 0.015 * accuracy,
        "ndcg@10": 0.019 * accuracy,
        "recall@20": 0.024 * accuracy,
        "ndcg@20": 0.021 * accuracy,
        "recall@50": 0.044 * accuracy,
        "ndcg@50": 0.028 * accuracy,
        "price_purchase_amount_weighted_hit@10": economic,
        "vndcg@10": economic / 36,
    }


def test_attribution_pass_requires_baseline_shuffle_accuracy_and_real_intervention():
    rows = {
        control.M1_MODEL_ID: _metrics(0.380),
        control.M4_ACTUAL_MODEL_ID: _metrics(0.390),
        control.M4_SHUFFLED_MODEL_ID: _metrics(0.385),
    }
    cvs = {
        control.M4_ACTUAL_MODEL_ID: 0.11,
        control.M4_SHUFFLED_MODEL_ID: 0.10,
    }

    passed = control.attribution_reading(
        rows,
        actual_vs_shuffle_top10_change_share=0.2,
        row_weight_cvs=cvs,
    )
    assert passed["attribution_pass"] is True
    assert passed["classification"] == "q_c_assignment_supported"

    rows[control.M4_SHUFFLED_MODEL_ID] = _metrics(0.391)
    failed = control.attribution_reading(
        rows,
        actual_vs_shuffle_top10_change_share=0.2,
        row_weight_cvs=cvs,
    )
    assert failed["attribution_pass"] is False


def test_colab_runs_the_locked_three_arm_screen():
    notebook_path = Path(
        "clv_m4_personalized_positive_weight_k1_assignment_control_"
        "dunnhumby_colab.ipynb"
    )
    if not notebook_path.exists():
        return
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("run_m4_k1_assignment_control_screen(cfg)") == 1
    assert "cfg.negative_count == 1" in source
    assert "summary['trained_models'] == list(controls.MODEL_IDS)" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "TO_BE_PINNED" not in source
