from dataclasses import replace

import numpy as np
import pytest

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


def test_fast_two_dataset_suite_runs_hm60_then_full_dunnhumby(monkeypatch):
    seen = []

    def fake_run(cfg):
        seen.append((cfg.dataset, cfg.window_days, cfg.gate_shape))
        return cfg.dataset

    monkeypatch.setattr(joint, "run_experiment", fake_run)
    result = joint.run_two_dataset_screening()

    assert seen == [("hm", 60, "high"), ("dunnhumby", None, "equal")]
    assert result == {"hm_w60": "hm", "dunnhumby_full": "dunnhumby"}


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


def test_user_axes_are_deterministic_historical_clv_components():
    values = np.array(
        [
            [0.2, 0.4, 0.6, 0.8, 1.0],
            [0.8, 0.6, 0.4, 0.2, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        np.float32,
    )
    valid = np.array([True, True, True])
    axes = joint.build_user_axis_inputs(values, valid)

    np.testing.assert_array_equal(axes["activity"], values[:, :3])
    np.testing.assert_array_equal(axes["value"], values[:, 3:])
    np.testing.assert_allclose(axes["n_hat"], [0.4, 0.6, 0.5])
    np.testing.assert_allclose(axes["v_hat"], [0.9, 0.1, 0.5])
    np.testing.assert_allclose(axes["clv_proxy"], axes["n_hat"] * axes["v_hat"])
    assert axes["q_n"].shape == (3,)
    assert axes["q_v"].shape == (3,)


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
