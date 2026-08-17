from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import lightgcn_clv_residual as residual
import lightgcn_clv_joint_nv as joint


def test_colab_drops_stale_repo_modules_before_importing_runner():
    notebook = json.loads(
        Path("clv_joint_nv_two_dataset_colab.ipynb").read_text(encoding="utf-8")
    )
    import_cell = "".join(notebook["cells"][2]["source"])

    assert "importlib.invalidate_caches()" in import_cell
    assert "sys.modules" in import_cell
    assert "str(repo)" in import_cell
    assert import_cell.index("sys.modules") < import_cell.index(
        "from lightgcn_clv_joint_nv import"
    )


def test_anchored_dunnhumby_colab_is_pinned_and_runs_the_fast_preset_once():
    notebook = json.loads(
        Path("clv_joint_nv_anchored_dunnhumby_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "e3865fafdffd02c04409c3eaed20d1743c159d33" in source
    assert "configure_anchored_dunnhumby_run" in source
    assert source.count("result_df = run_experiment(cfg)") == 1
    assert "ACKNOWLEDGE_HIGH_COST" not in source


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
    assert summary["gamma"] == {
        "initial_score_strength": 0.01,
        "application": "sqrt(gamma) applied symmetrically to user and item N/V blocks",
    }
    assert summary["gate_source"] == {
        "q_n": "train-history percentile of repeat transactions / customer age",
        "q_v": "train-history percentile of mean transaction value",
    }
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


def test_fast_anchored_dunnhumby_preset_runs_only_m1_and_m2_on_validation():
    cfg = joint.configure_anchored_dunnhumby_run()
    summary = joint.preflight_summary(cfg)

    assert cfg.dataset == "dunnhumby"
    assert cfg.seed == 42
    assert cfg.window_days is None
    assert cfg.gate_shape == "equal"
    assert cfg.anchor_weight == 0.5
    assert cfg.compute_variable_validity is False
    assert summary["models"] == ["m1", "joint_nv_anchored"]
    assert summary["loss"] == {
        "type": "preference_anchored_bpr",
        "full_weight": 0.5,
        "id_anchor_weight": 0.5,
    }
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["variable_validity"] is None
    assert summary["variable_validity_source"] is None


def test_matching_result_payload_uses_anchored_model_identity(tmp_path):
    checkpoint = tmp_path / "joint_nv_anchored_dunnhumby_s42_x.pt"
    checkpoint.touch()
    result_path = tmp_path / "m2_joint_nv_dunnhumby_x.json"
    result_path.write_text(
        json.dumps({"checkpoints": {"joint_nv_anchored": str(checkpoint)}}),
        encoding="utf-8",
    )

    payload = joint._matching_result_payload(
        tmp_path, checkpoint, model_id="joint_nv_anchored"
    )

    assert payload is not None
    assert payload["_result_json"] == str(result_path)


def test_block_comparison_separates_id_recovery_and_axis_increment():
    rows = joint._block_comparison_rows(
        {
            "id_only": {"revenue@10": 8.0},
            "id_n": {"revenue@10": 9.0},
            "id_v": {"revenue@10": 11.0},
            "full": {"revenue@10": 12.0},
        },
        {"revenue@10": 10.0},
    )

    by_view = {row["view"]: row for row in rows}
    assert by_view["id_only"]["delta_vs_m1"] == -2.0
    assert by_view["id_n"]["delta_vs_id_only"] == 1.0
    assert by_view["id_v"]["delta_vs_id_only"] == 3.0
    assert by_view["full"]["delta_vs_m1"] == 2.0
    assert by_view["full"]["relative_change_vs_m1_pct"] == 20.0


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
    assert not axes["activity_valid"][1]
    assert axes["activity_valid"][[0, 2, 3]].all()
    assert not axes["value_valid"][1]
    assert axes["value_valid"][[0, 2, 3]].all()
    np.testing.assert_allclose(
        axes["clv_proxy"],
        axes["n_behavior_score"] * axes["v_behavior_score"],
    )
    assert axes["q_n"].shape == (5,)
    assert axes["q_v"].shape == (5,)
    np.testing.assert_allclose(
        axes["q_n"][[0, 2, 3]], [5 / 6, 1 / 6, 3 / 6]
    )
    np.testing.assert_allclose(
        axes["q_v"][[0, 2, 3]], [1 / 6, 5 / 6, 3 / 6]
    )


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
