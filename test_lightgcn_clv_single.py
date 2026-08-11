import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@dataclass
class ReuseFixture:
    result_json: Path
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
    checkpoint = tmp_path / "single_adapter.pt"
    torch.save({"ev_all": ev_all}, checkpoint)
    rows = {float(lam): _reuse_metric_row(float(lam)) for lam in cfg.lambda_eval}
    payload = {
        "source_revision": "legacy-revision",
        "input_manifest": manifest,
        "config": asdict(cfg),
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
        "user_profile": SimpleNamespace(feature_names=("u0",)),
        "item_profile": SimpleNamespace(numeric_names=("i0",)),
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
        current_manifest=manifest,
        base_hash="base-state",
        cfg=cfg,
        base_cfg={"K_LIST": [10, 20, 50]},
        context=context,
        data={"n_items": 2},
        rows_by_lambda=rows,
    )


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


def test_reuse_rejects_input_manifest_mismatch(reuse_fixture):
    import lightgcn_clv_single as single

    fixture = reuse_fixture
    changed = fixture.current_manifest | {
        "transactions": {"path": "/x", "bytes": 1, "sha256": "changed"}
    }
    with pytest.raises(RuntimeError, match="input manifest"):
        single.load_reusable_single_full(
            fixture.result_json,
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
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash="wrong",
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )


def test_reuse_accepts_exact_legacy_full_and_relabels_rows(reuse_fixture):
    import lightgcn_clv_single as single

    reused = single.load_reusable_single_full(
        reuse_fixture.result_json,
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
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )
