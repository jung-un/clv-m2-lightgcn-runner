import json
from pathlib import Path

import numpy as np
import pytest


def _payload(**config_overrides):
    return {
        "code_version": "clv-dual-axis-fixed-v1.1",
        "source_revision": "abc",
        "result_fingerprint": "fixture",
        "input_manifest": {},
        "config": {
            "dataset": "dunnhumby",
            "seed_list": [42],
            "window_days": None,
            "input_days": 365,
            "target_days": 90,
            "anchor_offsets": [270, 180, 90],
            "eval_test": False,
            "eval_holdout": False,
            **config_overrides,
        },
        "models": [
            "m1",
            "dual_clv_fixed",
            "dual_shuffled_user",
            "dual_adapter_only",
        ],
        "selected_operating_point": {"gate_shape": "equal", "lambda": 2.0},
        "checkpoint_paths": {
            "m1_s42": "/missing/m1.pt",
            "encoder_s42": "/missing/encoder.pt",
            "dual_clv_fixed_s42": "/missing/dual.pt",
        },
        "checkpoint_sha256": {
            "m1_s42": "0" * 64,
            "encoder_s42": "1" * 64,
            "dual_clv_fixed_s42": "2" * 64,
        },
    }


def test_checkpoint_diagnostic_rejects_non_validation_before_data(monkeypatch, tmp_path):
    import lightgcn_clv_dual_checkpoint_diagnostic as diagnostic

    path = tmp_path / "result.json"
    path.write_text(json.dumps(_payload(eval_test=True)))
    monkeypatch.setattr(
        diagnostic.v3,
        "prepare_data",
        lambda *_: (_ for _ in ()).throw(AssertionError("data touched")),
    )
    with pytest.raises(ValueError, match="validation-only"):
        diagnostic.run_checkpoint_diagnostic(path)


def test_checkpoint_diagnostic_checks_every_checkpoint_hash(tmp_path):
    import lightgcn_clv_dual_checkpoint_diagnostic as diagnostic

    paths, hashes = {}, {}
    for name in ("m1_s42", "encoder_s42", "dual_clv_fixed_s42"):
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes(name.encode())
        paths[name] = str(checkpoint)
        hashes[name] = "f" * 64
    payload = _payload()
    payload["checkpoint_paths"], payload["checkpoint_sha256"] = paths, hashes
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="checkpoint hash"):
        diagnostic.run_checkpoint_diagnostic(path)


def test_quadrants_use_train_only_percentiles_and_report_paired_gain():
    from lightgcn_clv_dual_checkpoint_diagnostic import quadrant_metrics

    baseline = {
        "recall": np.array([1.0, 1.0, 1.0, 1.0]),
        "ndcg": np.array([1.0, 1.0, 1.0, 1.0]),
        "revenue": np.array([1.0, 1.0, 1.0, 1.0]),
        "arp": np.array([0.2, 0.2, 0.2, 0.2]),
    }
    model = {
        "recall": np.array([2.0, 3.0, 4.0, 5.0]),
        "ndcg": np.array([2.0, 3.0, 4.0, 5.0]),
        "revenue": np.array([2.0, 3.0, 4.0, 5.0]),
        "arp": np.array([0.3, 0.3, 0.3, 0.3]),
    }
    table = quadrant_metrics(
        q_n=np.array([0.2, 0.8, 0.2, 0.8]),
        q_v=np.array([0.2, 0.2, 0.8, 0.8]),
        valid=np.ones(4, bool),
        eval_users=np.arange(4),
        model_per_user=model,
        baseline_per_user=baseline,
        model_id="n_plus_v",
        lam=1.0,
        n_boot=100,
    )
    revenue = table.loc[table.metric.eq("revenue")].set_index("quadrant")
    assert set(revenue.index) == {"low_low", "activity", "value", "core"}
    assert revenue.loc["activity", "user_count"] == 1
    assert revenue.loc["core", "mean_delta"] == pytest.approx(4.0)
    assert revenue.loc["core", "improved_user_share"] == 1.0


def test_dataset_diagnostic_grids_are_fixed_around_existing_selection():
    from lightgcn_clv_dual_checkpoint_diagnostic import diagnostic_spec

    assert diagnostic_spec("dunnhumby", None) == {
        "gate_shape": "equal",
        "lambdas": (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0),
    }
    assert diagnostic_spec("hm", 60) == {
        "gate_shape": "high",
        "lambdas": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
    }


def test_colab_runs_both_checkpoint_diagnostics_without_training():
    notebook = json.loads(Path("clv_dual_checkpoint_diagnostic_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_checkpoint_diagnostic" in source
    assert "run_experiment" not in source
    assert "train_clv" not in source
    assert "results_clv_dual_dunnhumby" in source
    assert "results_clv_dual_hm_w60" in source
