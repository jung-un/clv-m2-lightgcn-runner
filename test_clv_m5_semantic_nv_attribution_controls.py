import json
from pathlib import Path

import numpy as np

import lightgcn_clv_m5_semantic_nv_attribution_controls as runner


def _prepared():
    return {
        "q_n": np.array([0.1, 0.3, 0.8, 0.9], dtype=np.float32),
        "q_v": np.array([0.2, 0.4, 0.7, 0.8], dtype=np.float32),
        "q_c": np.array([0.1, 0.4, 0.7, 0.95], dtype=np.float32),
        "clv_valid": np.ones(4, dtype=bool),
        "user_activity_gate": np.array([0.1, 0.3, 0.8, 0.9], dtype=np.float32),
        "user_economic_input": np.arange(20, dtype=np.float32).reshape(4, 5),
        "user_economic_valid": np.ones(4, dtype=bool),
        "user_bin_fit": np.arange(16, dtype=np.float32).reshape(4, 4),
        "degree_bin": np.array([0, 0, 1, 1], dtype=np.int64),
        "degree_percentile": np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
    }


def test_preflight_trains_only_two_attribution_controls(tmp_path):
    cfg = runner.configure_semantic_nv_attribution_controls(
        out_dir=str(tmp_path / "out"),
        actual_m5_result_json=str(tmp_path / "actual.json"),
        m4_control_result_json=str(tmp_path / "m4.json"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        runner.JOINT_SHUFFLE_MODEL_ID,
        runner.DEGREE_CONTROL_MODEL_ID,
    ]
    assert summary["reused_models"] == [runner.M4_MODEL_ID, runner.M5_MODEL_ID]
    assert summary["fixed"]["rho"] == 0.15
    assert summary["fixed"]["beta"] == 0.25
    assert summary["fixed"]["positive_weight_lambda"] == 0.5
    assert summary["reading_rule"]["primary_metrics"] == list(
        runner.PRIMARY_METRICS
    )


def test_joint_shuffle_moves_complete_semantic_and_loss_tuple():
    prepared = _prepared()
    cfg = runner.configure_semantic_nv_attribution_controls(
        shuffle_degree_bins=10
    )
    runner.attach_control_assignments(prepared, cfg)
    shuffled = prepared["joint_shuffle"]

    for target, source in enumerate(shuffled["source_user"]):
        assert prepared["degree_bin"][target] == prepared["degree_bin"][source]
        assert shuffled["q_n"][target] == prepared["q_n"][source]
        assert shuffled["q_c"][target] == prepared["q_c"][source]
        np.testing.assert_array_equal(
            shuffled["user_economic_input"][target],
            prepared["user_economic_input"][source],
        )
        np.testing.assert_array_equal(
            shuffled["user_bin_fit"][target], prepared["user_bin_fit"][source]
        )


def test_degree_control_changes_only_m4_q_c_gate():
    prepared = _prepared()
    cfg = runner.configure_semantic_nv_attribution_controls()
    runner.attach_control_assignments(prepared, cfg)
    degree = prepared["degree_control"]

    np.testing.assert_array_equal(degree["q_c"], prepared["degree_percentile"])
    np.testing.assert_array_equal(degree["q_n"], prepared["q_n"])
    np.testing.assert_array_equal(
        degree["user_economic_input"], prepared["user_economic_input"]
    )
    np.testing.assert_array_equal(degree["user_bin_fit"], prepared["user_bin_fit"])


def _metrics(primary=0.2):
    return {
        "recall@10": 0.10,
        "ndcg@10": 0.11,
        "recall@20": 0.20,
        "ndcg@20": 0.21,
        "recall@50": 0.30,
        "ndcg@50": 0.31,
        "vndcg@10": primary,
        "price_purchase_amount_weighted_hit@10": primary + 0.1,
    }


def test_attribution_rule_requires_actual_to_beat_both_controls_on_both_metrics():
    rows = {
        runner.M4_MODEL_ID: _metrics(0.20),
        runner.M5_MODEL_ID: _metrics(0.23),
        runner.JOINT_SHUFFLE_MODEL_ID: _metrics(0.21),
        runner.DEGREE_CONTROL_MODEL_ID: _metrics(0.22),
    }
    positive = runner.attribution_reading(rows)
    rows[runner.DEGREE_CONTROL_MODEL_ID][
        "price_purchase_amount_weighted_hit@10"
    ] = rows[runner.M5_MODEL_ID]["price_purchase_amount_weighted_hit@10"]
    nonpositive = runner.attribution_reading(rows)

    assert positive["positive_attribution_screen"] is True
    assert positive["joint_assignment_signal"] is True
    assert positive["degree_control_signal"] is True
    assert nonpositive["positive_attribution_screen"] is False
    assert nonpositive["degree_control_signal"] is False
    assert nonpositive["accuracy_guard_required"] is False


def test_colab_runs_attribution_controls_and_reuses_completed_results():
    notebook_path = Path(
        "clv_m5_semantic_nv_personalized_positive_dunnhumby_attribution_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "run_semantic_nv_attribution_controls(cfg)" in source
    assert "m5_semantic_nv_single_426685be4486.json" in source
    assert "m5_semantic_nv_m4_control_cea1a4ef5860.json" in source
    assert "JOINT_SHUFFLE_MODEL_ID" in source
    assert "DEGREE_CONTROL_MODEL_ID" in source
