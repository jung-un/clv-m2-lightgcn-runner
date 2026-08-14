import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_suite_config_is_hm_full_period_seed42_validation_only(tmp_path):
    import lightgcn_clv_dual_hm2y_suite as suite

    cfg = suite.configure_hm2y_suite(
        out_dir=tmp_path / "out", m1_checkpoint_dir=tmp_path / "m1"
    )

    assert cfg.dataset == "hm"
    assert cfg.window_days is None
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is cfg.eval_holdout is False
    assert suite.MODELS == (
        "m1",
        "dual_clv_fixed",
        "dual_shuffled_user",
        "dual_adapter_only",
    )
    assert suite.GATE_SHAPE == "high"
    assert suite.TARGET_RHO == pytest.approx(0.2)
    assert suite.BATCH_CANDIDATES == (131072, 65536, 32768)


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "dunnhumby"},
        {"window_days": 60},
        {"seed_list": (43,)},
        {"eval_test": True},
        {"eval_holdout": True},
    ],
)
def test_suite_rejects_protocol_changes(tmp_path, changes):
    import lightgcn_clv_dual_hm2y_suite as suite

    cfg = suite.configure_hm2y_suite(
        out_dir=tmp_path / "out", m1_checkpoint_dir=tmp_path / "m1"
    )
    with pytest.raises(ValueError):
        suite.validate_suite_config(dataclasses.replace(cfg, **changes))


def test_choose_batch_size_uses_first_passing_candidate():
    import lightgcn_clv_dual_hm2y_suite as suite

    calls = []

    def probe(batch_size):
        calls.append(batch_size)
        if batch_size == 131072:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return True

    chosen = suite.choose_batch_size(suite.BATCH_CANDIDATES, probe)

    assert chosen == 65536
    assert calls == [131072, 65536]


def test_choose_batch_size_does_not_hide_non_oom_error():
    import lightgcn_clv_dual_hm2y_suite as suite

    with pytest.raises(RuntimeError, match="real bug"):
        suite.choose_batch_size((131072,), lambda _batch: (_ for _ in ()).throw(
            RuntimeError("real bug")
        ))


def test_manifest_skips_completed_stage_and_rejects_identity_change(tmp_path):
    import lightgcn_clv_dual_hm2y_suite as suite

    identity = suite.suite_identity("cfg", "source", "input")
    manifest = suite.SuiteManifest.open(tmp_path, identity)
    assert manifest.is_completed("dual_clv_fixed") is False

    manifest.start("dual_clv_fixed")
    manifest.complete("dual_clv_fixed", checkpoint="model.pt", sha256="abc")
    reloaded = suite.SuiteManifest.open(tmp_path, identity)
    assert reloaded.is_completed("dual_clv_fixed") is True
    assert reloaded.stage("dual_clv_fixed")["sha256"] == "abc"

    changed = suite.suite_identity("changed", "source", "input")
    with pytest.raises(RuntimeError, match="identity"):
        suite.SuiteManifest.open(tmp_path, changed)


def test_suite_decision_requires_m1_and_both_control_wins():
    import lightgcn_clv_dual_hm2y_suite as suite

    baseline = {
        **{
            f"{metric}@{k}": 1.0
            for metric in ("recall", "ndcg")
            for k in (10, 20, 50)
        },
        "revenue@10": 2.0,
    }
    fixed = {**baseline, "revenue@10": 2.2}
    controls = {
        "dual_shuffled_user": {**baseline, "revenue@10": 2.1},
        "dual_adapter_only": {**baseline, "revenue@10": 2.05},
    }
    assert suite.suite_decision(baseline, fixed, controls)["success"] is True

    controls["dual_adapter_only"]["revenue@10"] = 2.3
    decision = suite.suite_decision(baseline, fixed, controls)
    assert decision["success"] is False
    assert "dual_adapter_only" in decision["failed_controls"]


def test_read_progress_uses_drive_file_only(tmp_path):
    import lightgcn_clv_dual_hm2y_suite as suite

    root = suite.suite_root(tmp_path)
    root.mkdir(parents=True)
    (root / "progress.json").write_text(
        json.dumps({"stage": "m1", "epoch": 4, "status": "running"})
    )

    assert suite.read_progress(tmp_path) == {
        "stage": "m1",
        "epoch": 4,
        "status": "running",
    }


def test_dual_prepare_accepts_batch_selector_and_resume_stores():
    source = Path("lightgcn_clv_dual.py").read_text()
    assert "batch_selector=None" in source
    assert "progress_stores=None" in source


def test_suite_runs_three_variants_once_then_loads_completed_stages(
    monkeypatch, tmp_path
):
    import lightgcn_clv_dual_hm2y_suite as suite

    cfg = suite.configure_hm2y_suite(
        out_dir=tmp_path / "out", m1_checkpoint_dir=tmp_path / "m1"
    )
    monkeypatch.setattr(suite.moe, "build_input_manifest", lambda _schema: {"x": 1})
    monkeypatch.setattr(suite.moe, "manifest_hash", lambda _manifest: "input")
    monkeypatch.setattr(suite.moe, "source_revision", lambda: "source")
    monkeypatch.setattr(suite, "_m1_batch_probe", lambda *_args: True)
    m1_path = tmp_path / "m1.pt"
    encoder_path = tmp_path / "encoder.pt"
    m1_path.write_bytes(b"m1")
    encoder_path.write_bytes(b"encoder")
    baseline = {
        **{
            f"{metric}@{k}": 1.0
            for metric in ("recall", "ndcg")
            for k in (10, 20, 50)
        },
        "revenue@10": 2.0,
    }

    def fake_prepare(_cfg, **kwargs):
        assert kwargs["batch_selector"](
            {"n_users": 2, "n_items": 3}, {"BATCH_SIZE": 8192}
        ) == 131072
        return {
            "data": {"n_users": 2, "n_items": 3},
            "m1_checkpoint": str(m1_path),
            "encoder_checkpoint": str(encoder_path),
            "baseline_flat": baseline,
            "baseline_per_user": {
                metric: np.zeros(2) for metric in ("recall", "ndcg", "revenue", "arp")
            },
            "base_cfg": {"BATCH_SIZE": 131072, "N_BOOT": 10},
            "fingerprint": "fp",
            "manifest": {"x": 1},
            "revision": "source",
        }

    monkeypatch.setattr(suite.dual, "_prepare", fake_prepare)
    monkeypatch.setattr(suite.dual, "_fresh_base", lambda *_args, **_kwargs: object())
    trained = []

    def fake_train(model_id, *_args, **_kwargs):
        trained.append(model_id)
        checkpoint = tmp_path / f"{model_id}.pt"
        checkpoint.write_bytes(model_id.encode())
        return {
            "model": object(),
            "training": {"epochs_run": 1},
            "diagnostics": {},
            "checkpoint": str(checkpoint),
        }

    monkeypatch.setattr(suite.dual, "_train_variant", fake_train)
    monkeypatch.setattr(
        suite,
        "_load_variant",
        lambda model_id, *_args: {
            "model": object(),
            "training": {"loaded": True},
            "diagnostics": {},
            "checkpoint": str(tmp_path / f"{model_id}.pt"),
        },
    )
    economics = {
        "dual_clv_fixed": 2.2,
        "dual_shuffled_user": 2.1,
        "dual_adapter_only": 2.05,
    }
    monkeypatch.setattr(
        suite,
        "_evaluate_variant",
        lambda run, _prepared: (
            {**baseline, "lambda": 1.0, "revenue@10": economics[
                next(name for name in economics if name in run["checkpoint"])
            ]},
            {metric: np.zeros(2) for metric in ("recall", "ndcg", "revenue", "arp")},
        ),
    )

    def fake_persist(*_args):
        frame = pd.DataFrame()
        frame.attrs["result_paths"] = {"json": str(tmp_path / "result.json")}
        return frame

    monkeypatch.setattr(suite, "_persist", fake_persist)

    suite.run_hm2y_suite(cfg)
    suite.run_hm2y_suite(cfg)

    assert trained == list(suite.MODELS[1:])
