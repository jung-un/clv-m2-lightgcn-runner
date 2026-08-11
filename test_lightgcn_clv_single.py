import dataclasses
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@dataclass
class ReuseFixture:
    result_json: Path
    checkpoint_sha256: str
    current_manifest: dict
    base_hash: str
    cfg: object
    base_cfg: dict
    context: dict
    data: dict
    rows_by_lambda: dict


def _reuse_metric_row(lam):
    row = {
        "seed": 42,
        "model_id": "single_adapter",
        "split": "val",
        "lambda": lam,
        "role": "control",
        "revenue@10": 1.0 + 0.01 * lam,
        "arp@10": 0.2,
    }
    for k in (10, 20, 50):
        row[f"recall@{k}"] = 0.1
        row[f"ndcg@{k}"] = 0.1
        row[f"n_distinct@{k}"] = 3
        row[f"exposure_entropy@{k}"] = 1.0
        row[f"eff_catalog@{k}"] = 2.7
        row[f"top10_share@{k}"] = 0.5
        row[f"top100_share@{k}"] = 1.0
    return row


@pytest.fixture
def reuse_fixture(tmp_path, monkeypatch):
    import lightgcn_clv_moe as moe
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    manifest = {
        "transactions": {"path": "/tx", "bytes": 2, "sha256": "aa"},
        "item_metadata": {"path": "/item", "bytes": 2, "sha256": "bb"},
    }
    ev_all = np.array([1.0, 2.0], dtype=np.float32)
    user_values = np.array([[0.5], [1.5]], dtype=np.float32)
    user_valid = np.array([True, True])
    item_numeric = np.array([[0.2], [0.8]], dtype=np.float32)
    item_category_ids = np.array([1, 2], dtype=np.int64)
    item_valid = np.array([True, True])
    checkpoint = tmp_path / "single_adapter.pt"
    torch.save(
        {
            "ev_all": ev_all,
            "user_profile": user_values,
            "user_valid": user_valid,
            "user_feature_names": ("u0",),
            "item_numeric": item_numeric,
            "item_category_ids": item_category_ids,
            "item_valid": item_valid,
            "item_numeric_names": ("i0",),
            "item_n_categories": 3,
        },
        checkpoint,
    )
    base_cfg = {
        "DIM": 64,
        "N_LAYERS": 3,
        "BATCH_SIZE": 1024,
        "EPOCHS": 100,
        "EARLY_STOP": 20,
        "LR": 0.001,
        "REG_MODE": "layer0",
        "PREF_REG": 0.0001,
        "WD": 0.0,
        "NEG_MODE": "uniform",
        "WINDOW_DAYS": None,
        "VAL_DAYS": 30,
        "TEST_DAYS": 30,
        "HOLDOUT_DAYS": 30,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "K_LIST": [10, 20, 50],
        "SEG_EDGES": [0.2, 0.8],
        "EVAL_BATCH": 256,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "N_BOOT": 10,
    }
    rows = {float(lam): _reuse_metric_row(float(lam)) for lam in cfg.lambda_eval}
    payload = {
        "source_revision": "legacy-revision",
        "input_manifest": manifest,
        "config": asdict(cfg),
        "base_config": base_cfg,
        "baseline_state_hashes": {"42": "base-state"},
        "feature_schema": {
            "user": ["u0"],
            "item_numeric": ["i0"],
        },
        "checkpoint_paths": {"single_adapter_s42": str(checkpoint)},
        "absolute_rows": list(rows.values()),
        "training": {"single_adapter_s42": {"base_updates_at_best": 3}},
        "moe_diagnostics": {
            "single_adapter_s42": {"parameter_match_ratio": 1.0}
        },
    }
    result_json = tmp_path / "legacy.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    context = {
        "artifact": SimpleNamespace(ev_all=ev_all),
        "user_profile": SimpleNamespace(
            values=user_values,
            valid_user=user_valid,
            feature_names=("u0",),
        ),
        "item_profile": SimpleNamespace(
            numeric=item_numeric,
            category_ids=item_category_ids,
            valid_item=item_valid,
            numeric_names=("i0",),
            n_categories=3,
        ),
        "caches": {"val": object()},
    }
    monkeypatch.setattr(moe, "load_moe_checkpoint", lambda *args, **kwargs: object())

    def fake_flat(model, lam, *args, **kwargs):
        row = rows[float(lam)]
        metrics = {
            key: value
            for key, value in row.items()
            if key not in {"seed", "model_id", "split", "lambda", "role"}
        }
        return metrics, None

    monkeypatch.setattr(moe, "_flat_evaluation", fake_flat)
    return ReuseFixture(
        result_json=result_json,
        checkpoint_sha256=moe.file_sha256(checkpoint),
        current_manifest=manifest,
        base_hash="base-state",
        cfg=cfg,
        base_cfg=base_cfg,
        context=context,
        data={"n_items": 2},
        rows_by_lambda=rows,
    )


def _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue):
    import lightgcn_clv_single as single

    calls = {"controls": []}
    baseline = _reuse_metric_row(0.0) | {
        "model_id": "m1",
        "role": "baseline",
        "revenue@10": 1.0,
    }
    prepared = SimpleNamespace(
        out_dir=tmp_path,
        baseline_row=baseline,
        baseline_metrics=baseline,
        baseline_per_user={
            "recall": np.zeros(2),
            "ndcg": np.zeros(2),
            "revenue": np.zeros(2),
            "arp": np.zeros(2),
        },
        input_manifest={"transactions": {}, "item_metadata": {}},
        baseline_state_hash="base-state",
        base_cfg={"N_BOOT": 10, "K_LIST": [10, 20, 50]},
        data={"data_stats": {}},
        context={
            "user_profile": SimpleNamespace(feature_names=("u0",)),
            "item_profile": SimpleNamespace(numeric_names=("i0",)),
            "artifact": SimpleNamespace(diagnostics={}),
        },
        source_revision="test-revision",
        anchor_diagnostics=(
            {"offset_days": 21, "n_users": 2},
            {"offset_days": 14, "n_users": 2},
            {"offset_days": 7, "n_users": 2},
        ),
    )
    monkeypatch.setattr(single, "_prepare_validation_context", lambda cfg: prepared)

    def fake_variant(prepared, cfg, model_id):
        calls["controls"].append(model_id)
        revenue = full_revenue if model_id == "single_full" else 1.01
        rows = []
        per_user = {}
        for lam in cfg.lambda_eval:
            row = _reuse_metric_row(float(lam)) | {
                "model_id": model_id,
                "role": "model" if model_id == "single_full" else "control",
                "revenue@10": revenue if lam == 1.0 else 1.0,
            }
            rows.append(row)
            per_user[float(lam)] = prepared.baseline_per_user
        checkpoint = tmp_path / f"{model_id}.pt"
        torch.save({"model_id": model_id}, checkpoint)
        return single.VariantRun(
            model_id=model_id,
            rows=tuple(rows),
            per_user=per_user,
            training={"base_updates_at_best": 3},
            diagnostics={
                "parameter_match_ratio": 1.0,
                "starting_base_state_hash": "base-state",
                "routed_profile_sha256": "a" * 64,
                "has_profile_sha256": "b" * 64,
                "adapter_parameter_count": 10,
                "joint_trainable_parameter_count": 20,
            },
            checkpoint=str(checkpoint),
            reuse_provenance=None,
        )

    monkeypatch.setattr(single, "_train_evaluate_variant", fake_variant)
    pref_row = baseline | {
        "model_id": "pref_continue",
        "role": "control",
        "lambda": 0.0,
        "revenue@10": 1.0,
    }
    monkeypatch.setattr(single, "_run_pref_continue", lambda *args, **kwargs: pref_row)
    return calls


def test_default_single_screening_is_seed42_validation_only():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    summary = single.preflight_summary(cfg)
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert summary["primary_model_id"] == "single_full"
    assert summary["required_controls"] == [
        "single_zero_user",
        "single_shuffled_user",
        "single_base_only",
    ]
    assert summary["mechanism_controls"] == ["single_zero_item"]
    assert summary["graph_mode"] == "binary"
    assert summary["loss_mode"] == "plain"


def test_hm_60day_preset_is_validation_only_and_fits_train_window():
    import lightgcn_clv_single as single

    cfg = single.configure_hm_60day_run()
    assert cfg.dataset == "hm"
    assert cfg.window_days == 60
    assert cfg.seed_list == (42,)
    assert (cfg.input_days, cfg.target_days) == (14, 7)
    assert cfg.anchor_offsets == (21, 14, 7)
    assert cfg.input_days + max(cfg.anchor_offsets) == 35
    assert not cfg.eval_test
    assert not cfg.eval_holdout


@pytest.mark.parametrize(
    ("key", "value"),
    [("window_days", 61), ("input_days", 28), ("eval_test", True)],
)
def test_hm_60day_preset_rejects_frozen_overrides(key, value):
    import lightgcn_clv_single as single

    with pytest.raises(ValueError, match="H&M 60-day"):
        single.configure_hm_60day_run(**{key: value})


def test_anchor_diagnostics_summarize_observation_and_future_targets():
    import lightgcn_clv_residual as residual
    import lightgcn_clv_single as single

    observed_days = residual.NUMERIC_FEATURES.index("observed_days")
    numeric = np.zeros((2, len(residual.NUMERIC_FEATURES)), dtype=np.float32)
    numeric[:, observed_days] = [4.0, 14.0]
    anchor = residual.AnchorExamples(
        offset_days=21,
        observation_start="2020-01-01",
        observation_end="2020-01-14",
        target_start="2020-01-15",
        target_end="2020-01-21",
        user_ids=np.array([3, 8]),
        numeric=numeric,
        valid=np.ones_like(numeric, dtype=bool),
        purchase_target=np.array([0.0, 1.0], dtype=np.float32),
        amount_target=np.array([0.0, 30.0], dtype=np.float32),
    )
    dataset = residual.AnchorDataset(
        anchors=[anchor], train_end="2020-01-21", n_users=10
    )

    assert single.summarize_anchor_dataset(dataset) == (
        {
            "offset_days": 21,
            "n_users": 2,
            "observation_start": "2020-01-01",
            "observation_end": "2020-01-14",
            "target_start": "2020-01-15",
            "target_end": "2020-01-21",
            "observed_days_p10": 5.0,
            "observed_days_median": 9.0,
            "observed_days_p90": 13.0,
            "purchase_rate": 0.5,
            "future_amount_mean": 15.0,
        },
    )


def test_lambda_diagnostics_store_effective_residual_strength():
    import lightgcn_clv_single as single

    diagnostics = {
        "gate_entropy_mean": 0.0,
        "residual_to_base_score_std": 0.2,
        "parameter_match_ratio": 1.0,
        "expert_usage_mean": [1.0],
    }
    columns = single._diagnostic_columns_for_lambda(diagnostics, 0.5)
    assert columns["residual_to_base_score_std"] == 0.2
    assert columns["effective_residual_to_base_score_std"] == 0.1


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_direct_dataclass_cannot_open_protected_splits(field):
    import lightgcn_clv_moe as moe
    import lightgcn_clv_single as single

    cfg = dataclasses.replace(moe.MoEConfig(), **{field: True})
    with pytest.raises(ValueError, match="screening-only"):
        single.validate_single_config(cfg)


def test_single_screening_decision_requires_full_to_beat_required_controls():
    import lightgcn_clv_single as single

    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
        "pref_continue": 0.0,
    }
    rows = [
        {
            "seed": 42,
            "split": "val",
            "model_id": model_id,
            "lambda": selected[model_id],
            "revenue@10": revenue,
        }
        for model_id, revenue in {
            "single_full": 1.10,
            "single_zero_user": 1.04,
            "single_shuffled_user": 1.03,
            "single_zero_item": 1.12,
            "single_base_only": 1.02,
            "pref_continue": 1.01,
        }.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is True
    assert decision["mechanism_comparison"]["single_zero_item"] == 1.12
    rows[1]["revenue@10"] = 1.11
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False
    assert decision["failed_controls"] == ["single_zero_user"]


def test_decision_tie_with_required_control_is_failure():
    import lightgcn_clv_single as single

    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
    }
    values = {
        "single_full": 1.10,
        "single_zero_user": 1.10,
        "single_shuffled_user": 1.03,
        "single_zero_item": 1.12,
        "single_base_only": 1.02,
    }
    rows = [
        {
            "seed": 42,
            "split": "val",
            "model_id": model_id,
            "lambda": selected[model_id],
            "revenue@10": revenue,
        }
        for model_id, revenue in values.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False


def test_validate_rejects_changed_lambda_grid():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    with pytest.raises(ValueError, match="lambda grid"):
        single.validate_single_config(
            dataclasses.replace(cfg, lambda_eval=(0.0, 1.0))
        )


def test_validate_rejects_changed_accuracy_tolerance():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    with pytest.raises(ValueError, match="accuracy tolerance"):
        single.validate_single_config(
            dataclasses.replace(cfg, accuracy_tolerance=0.02)
        )


@pytest.mark.parametrize(
    ("trusted_sha", "message"),
    [(None, "trusted checkpoint SHA"), ("0" * 64, "checkpoint SHA mismatch")],
)
def test_reuse_requires_and_verifies_trusted_checkpoint_sha(
    reuse_fixture, trusted_sha, message
):
    import lightgcn_clv_single as single

    with pytest.raises(RuntimeError, match=message):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            expected_checkpoint_sha256=trusted_sha,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


def test_reuse_rejects_input_manifest_mismatch(reuse_fixture):
    import lightgcn_clv_single as single

    fixture = reuse_fixture
    changed = fixture.current_manifest | {
        "transactions": {"path": "/x", "bytes": 1, "sha256": "changed"}
    }
    with pytest.raises(RuntimeError, match="input manifest"):
        single.load_reusable_single_full(
            fixture.result_json,
            expected_checkpoint_sha256=fixture.checkpoint_sha256,
            current_manifest=changed,
            baseline_state_hash=fixture.base_hash,
            cfg=fixture.cfg,
            base_cfg=fixture.base_cfg,
            context=fixture.context,
            data=fixture.data,
        )


def test_reuse_rejects_m1_state_or_feature_schema_mismatch(reuse_fixture):
    import lightgcn_clv_single as single

    with pytest.raises(RuntimeError, match="M1 state"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash="wrong",
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


def test_reuse_rejects_base_training_config_mismatch(reuse_fixture):
    import lightgcn_clv_single as single

    changed = reuse_fixture.base_cfg | {
        "BATCH_SIZE": reuse_fixture.base_cfg["BATCH_SIZE"] * 2
    }
    with pytest.raises(RuntimeError, match="base config"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=changed,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


@pytest.mark.parametrize(
    "checkpoint_key",
    [
        "user_profile",
        "user_valid",
        "item_numeric",
        "item_category_ids",
        "item_valid",
    ],
)
def test_reuse_rejects_checkpoint_input_or_mask_mismatch(
    reuse_fixture, checkpoint_key
):
    import lightgcn_clv_single as single

    payload = json.loads(reuse_fixture.result_json.read_text())
    checkpoint_path = Path(payload["checkpoint_paths"]["single_adapter_s42"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    changed = np.asarray(checkpoint[checkpoint_key]).copy()
    changed.reshape(-1)[0] = (
        not bool(changed.reshape(-1)[0])
        if changed.dtype == np.bool_
        else changed.reshape(-1)[0] + 1
    )
    checkpoint[checkpoint_key] = changed
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="checkpoint feature values"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            expected_checkpoint_sha256=single.moe.file_sha256(checkpoint_path),
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


def test_reuse_accepts_exact_legacy_full_and_relabels_rows(reuse_fixture):
    import lightgcn_clv_single as single

    reused = single.load_reusable_single_full(
        reuse_fixture.result_json,
        expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
        current_manifest=reuse_fixture.current_manifest,
        baseline_state_hash=reuse_fixture.base_hash,
        cfg=reuse_fixture.cfg,
        base_cfg=reuse_fixture.base_cfg,
        context=reuse_fixture.context,
        data=reuse_fixture.data,
    )
    assert {row["model_id"] for row in reused.rows} == {"single_full"}
    assert tuple(row["lambda"] for row in reused.rows) == reuse_fixture.cfg.lambda_eval
    assert reused.result_json_sha256


def test_reuse_treats_missing_legacy_window_as_full_window(reuse_fixture):
    import lightgcn_clv_single as single

    payload = json.loads(reuse_fixture.result_json.read_text())
    del payload["config"]["window_days"]
    reuse_fixture.result_json.write_text(json.dumps(payload))

    reused = single.load_reusable_single_full(
        reuse_fixture.result_json,
        expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
        current_manifest=reuse_fixture.current_manifest,
        baseline_state_hash=reuse_fixture.base_hash,
        cfg=reuse_fixture.cfg,
        base_cfg=reuse_fixture.base_cfg,
        context=reuse_fixture.context,
        data=reuse_fixture.data,
    )
    assert {row["model_id"] for row in reused.rows} == {"single_full"}


def test_reuse_rejects_full_window_result_for_short_window_config(reuse_fixture):
    import lightgcn_clv_single as single

    short_cfg = dataclasses.replace(reuse_fixture.cfg, window_days=60)
    with pytest.raises(RuntimeError, match="saved config mismatch for window_days"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=short_cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


def test_reuse_accepts_exact_short_window_result(reuse_fixture):
    import lightgcn_clv_single as single

    short_cfg = dataclasses.replace(reuse_fixture.cfg, window_days=60)
    short_base_cfg = reuse_fixture.base_cfg | {"WINDOW_DAYS": 60}
    payload = json.loads(reuse_fixture.result_json.read_text())
    payload["config"]["window_days"] = 60
    payload["base_config"]["WINDOW_DAYS"] = 60
    reuse_fixture.result_json.write_text(json.dumps(payload))

    reused = single.load_reusable_single_full(
        reuse_fixture.result_json,
        expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
        current_manifest=reuse_fixture.current_manifest,
        baseline_state_hash=reuse_fixture.base_hash,
        cfg=short_cfg,
        base_cfg=short_base_cfg,
        context=reuse_fixture.context,
        data=reuse_fixture.data,
    )
    assert {row["model_id"] for row in reused.rows} == {"single_full"}


def test_window_changes_result_fingerprint_and_m1_checkpoint_hash():
    import lightgcn_clv_moe as moe
    import lightgcn_clv_single as single
    import lightgcn_clv_v3 as v3

    full_cfg = single.configure_single_run("hm")
    short_cfg = single.configure_hm_60day_run()
    full_base_cfg = moe._pure_m1_config(full_cfg, "/tmp/m1-hm")
    short_base_cfg = moe._pure_m1_config(short_cfg, "/tmp/m1-hm")
    manifest = {"transactions": {"path": "/tx", "bytes": 2, "sha256": "aa"}}

    assert moe._result_fingerprint(full_cfg, full_base_cfg, manifest) != moe._result_fingerprint(
        short_cfg, short_base_cfg, manifest
    )
    assert v3.cfg_hash(full_base_cfg, v3.DCFG, "pref_only", 42) != v3.cfg_hash(
        short_base_cfg, v3.DCFG, "pref_only", 42
    )


def test_reuse_rejects_metric_round_trip_mismatch(reuse_fixture):
    import lightgcn_clv_single as single

    payload = json.loads(reuse_fixture.result_json.read_text())
    row = next(
        row
        for row in payload["absolute_rows"]
        if row["model_id"] == "single_adapter"
    )
    row["revenue@10"] += 0.01
    reuse_fixture.result_json.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="metric round-trip"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            expected_checkpoint_sha256=reuse_fixture.checkpoint_sha256,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


def test_runner_trains_full_then_all_controls_only_after_success(
    monkeypatch, tmp_path
):
    import lightgcn_clv_single as single

    calls = _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=1.10)
    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    frame = single.run_experiment(cfg)
    assert calls["controls"] == [
        "single_full",
        "single_zero_user",
        "single_shuffled_user",
        "single_base_only",
        "single_zero_item",
    ]
    assert set(frame.model_id) >= {
        "m1",
        "single_full",
        *single.REQUIRED_CONTROLS,
        *single.MECHANISM_CONTROLS,
        "pref_continue",
    }
    assert frame.attrs["screening_decision"]["success"] is True


def test_runner_stops_after_full_when_primary_selection_fails(monkeypatch, tmp_path):
    import lightgcn_clv_single as single

    calls = _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=0.99)
    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    frame = single.run_experiment(cfg)
    assert calls["controls"] == ["single_full"]
    assert set(frame.model_id) == {"m1", "single_full"}
    assert frame.attrs["screening_decision"]["success"] is False


def test_runner_persists_authoritative_json_and_exposure_metrics(
    monkeypatch, tmp_path
):
    import lightgcn_clv_single as single

    _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=1.10)
    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    frame = single.run_experiment(cfg)
    payload = json.loads(next(tmp_path.glob("clv_single_*.json")).read_text())
    assert [row["offset_days"] for row in payload["anchor_diagnostics"]] == [
        21,
        14,
        7,
    ]
    assert payload["screening_decision"] == frame.attrs["screening_decision"]
    assert (
        payload["variant_definitions"]["single_zero_user"]["user_profile"]
        == "zero"
    )
    assert {
        "n_distinct@10",
        "exposure_entropy@10",
        "eff_catalog@10",
        "top10_share@10",
        "top100_share@10",
    }.issubset(payload["absolute_rows"][0])
    full_revenue_delta_lambdas = {
        float(row["lambda"])
        for row in payload["delta"]
        if row["model_id"] == "single_full" and row["metric"] == "revenue"
    }
    assert full_revenue_delta_lambdas == set(cfg.lambda_eval)
    audit = payload["diagnostics"]["single_full_s42"]
    assert audit["starting_base_state_hash"] == "base-state"
    assert audit["adapter_parameter_count"] == 10
    assert len(audit["routed_profile_sha256"]) == 64
    assert set(payload["checkpoint_sha256"]) >= {
        "single_full_s42",
        "single_zero_user_s42",
        "single_shuffled_user_s42",
        "single_base_only_s42",
        "single_zero_item_s42",
    }
    assert all(len(value) == 64 for value in payload["checkpoint_sha256"].values())


def test_run_experiment_revalidates_before_data_access(monkeypatch):
    import lightgcn_clv_single as single

    monkeypatch.setattr(
        single,
        "_prepare_validation_context",
        lambda cfg: (_ for _ in ()).throw(AssertionError("data touched")),
    )
    bad = dataclasses.replace(
        single.configure_single_run("dunnhumby"), seed_list=(42, 43)
    )
    with pytest.raises(ValueError, match="seed 42"):
        single.run_experiment(bad)


def test_colab_has_pinned_source_preflight_gate_and_final_decision():
    notebook = json.loads(Path("clv_single_adapter_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "configure_single_run" in source
    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", source)
    assert "TO_BE_PINNED" not in source
    assert "preflight_summary" in source
    assert "reuse_full_result_json" in source
    assert "reuse_full_checkpoint_sha256" in source
    assert "ACKNOWLEDGE_HIGH_COST = False" in source
    assert "assert ACKNOWLEDGE_HIGH_COST" in source
    assert "screening_decision" in source
    assert "eval_test=False" in source and "eval_holdout=False" in source
