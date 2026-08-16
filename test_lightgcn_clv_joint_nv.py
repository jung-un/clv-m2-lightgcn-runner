from dataclasses import replace

import numpy as np
import pytest

import lightgcn_clv_residual as residual
import lightgcn_clv_joint_nv as joint


def test_hm60_preset_is_seed42_validation_only_and_plain_bpr():
    cfg = joint.configure_joint_nv_run("hm", short_hm=True)
    summary = joint.preflight_summary(cfg)

    assert cfg.seed == 42
    assert cfg.window_days == 60
    assert cfg.gate_shape == "high"
    assert summary["graph_mode"] == "binary"
    assert summary["negative_sampling"] == "uniform"
    assert summary["loss"] == "plain_bpr"
    assert summary["separate_encoder"] is False
    assert summary["frozen_or_external_base"] is False
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["models"] == ["m1", "joint_nv"]
    assert joint.variable_validity_plan(cfg) == {
        "input_days": 14,
        "target_days": 7,
        "anchor_offsets": (21, 14, 7),
    }


def test_fast_two_dataset_suite_runs_hm60_then_full_dunnhumby(monkeypatch):
    seen = []

    def fake_run(cfg):
        seen.append((cfg.dataset, cfg.window_days, cfg.gate_shape))
        return cfg.dataset

    monkeypatch.setattr(joint, "run_experiment", fake_run)
    result = joint.run_two_dataset_screening()

    assert seen == [("hm", 60, "high"), ("dunnhumby", None, "equal")]
    assert result == {"hm_w60": "hm", "dunnhumby_full": "dunnhumby"}


def test_dunnhumby_variable_validity_uses_only_internal_training_windows():
    cfg = joint.configure_joint_nv_run("dunnhumby", short_hm=False)
    assert joint.variable_validity_plan(cfg) == {
        "input_days": 365,
        "target_days": 90,
        "anchor_offsets": (270, 180, 90),
    }


def test_progress_paths_are_isolated_by_config_hash(tmp_path):
    cfg = joint.configure_joint_nv_run("hm", short_hm=True)
    old = joint._progress_store(
        tmp_path, "joint_nv", cfg, "old_config", "same_input", "old_source"
    )
    current = joint._progress_store(
        tmp_path, "joint_nv", cfg, "new_config", "same_input", "new_source"
    )

    assert old.root != current.root
    assert old.latest_checkpoint != current.latest_checkpoint


def test_public_runner_blocks_protected_splits_before_data_access(monkeypatch):
    cfg = replace(joint.configure_joint_nv_run("hm", short_hm=True), eval_test=True)
    touched = False

    def forbidden(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("prepare_data must not be called")

    monkeypatch.setattr(joint.v3, "prepare_data", forbidden)
    with pytest.raises(ValueError, match="validation-only"):
        joint.run_experiment(cfg)
    assert touched is False


def test_user_axes_use_literature_grounded_current_features_and_masks():
    numeric = np.zeros((3, len(residual.NUMERIC_FEATURES)), np.float32)
    valid = np.ones_like(numeric, dtype=bool)
    values = {
        "recency_days": [2, 20, 8],
        "basket_count": [8, 2, 5],
        "observed_days": [30, 10, 20],
        "gap_mean": [4, 0, 7],
        "avg_basket_value": [10, 100, 40],
    }
    for name, column in values.items():
        numeric[:, residual.NUMERIC_FEATURES.index(name)] = column
    valid[1, residual.NUMERIC_FEATURES.index("gap_mean")] = False
    snapshot = residual.AnchorExamples(
        0, 0, 1, 2, 1, np.array([0, 2, 3]), numeric, valid,
        np.zeros(3, np.float32), np.zeros(3, np.float32),
    )
    axes = joint.build_user_axis_inputs(snapshot, n_users=5)

    assert axes["activity"].shape == (5, 8)
    assert axes["value"].shape == (5, 2)
    assert axes["activity_names"] == (
        "repeat_transaction_count",
        "transaction_recency",
        "customer_age",
        "mean_transaction_gap",
        "valid_repeat_transaction_count",
        "valid_transaction_recency",
        "valid_customer_age",
        "valid_mean_transaction_gap",
    )
    assert axes["value_names"] == (
        "mean_transaction_value",
        "valid_mean_transaction_value",
    )
    assert "premium_share" not in axes["value_names"]
    assert axes["repeat_transaction_count"][0] == 7
    assert axes["repeat_transaction_count"][2] == 1
    assert axes["transaction_recency"][0] == 27
    assert axes["customer_age"][0] == 29
    assert axes["mean_transaction_value"][2] == 100
    assert axes["activity"][2, 7] == 0.0
    assert not axes["valid_user"][1]
    assert axes["valid_user"][[0, 2, 3]].all()
    np.testing.assert_allclose(
        axes["clv_proxy"],
        axes["n_behavior_score"] * axes["v_behavior_score"],
    )
    assert axes["q_n"].shape == (5,)
    assert axes["q_v"].shape == (5,)


def test_result_row_keeps_model_identity_and_full_metric_payload():
    metrics = {"recall@10": 0.1, "ndcg@10": 0.2, "revenue@10": 0.3}
    row = joint.result_row("joint_nv", "model", "high", 42, metrics, {"gamma_n": 0.4})
    assert row == {
        "seed": 42,
        "model_id": "joint_nv",
        "split": "val",
        "gate_shape": "high",
        "role": "model",
        "gamma_n": 0.4,
        **metrics,
    }
