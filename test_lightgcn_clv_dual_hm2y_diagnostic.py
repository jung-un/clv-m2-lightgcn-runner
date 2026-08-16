import dataclasses
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pytest


def _write_suite_payload(tmp_path):
    import lightgcn_clv_dual_hm2y_suite as suite

    checkpoints = {}
    hashes = {}
    for name in ("m1", "encoder", "dual_clv_fixed", "dual_shuffled_user",
                 "dual_adapter_only"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        checkpoints[name] = str(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = suite.configure_hm2y_suite(
        out_dir=tmp_path / "out",
        m1_checkpoint_dir=tmp_path / "m1-root",
    )
    payload = {
        "code_version": suite.CODE_VERSION,
        "result_fingerprint": "fp",
        "config": dataclasses.asdict(cfg),
        "models": list(suite.MODELS),
        "checkpoint_paths": checkpoints,
        "checkpoint_sha256": hashes,
        "interpretation": {"revenue": "price/purchase-amount weighted hit"},
    }
    result = tmp_path / "suite.json"
    result.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return result, payload


def test_hm2y_diagnostic_is_validation_only_and_uses_the_approved_grid(tmp_path):
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    cfg = diagnostic.configure_hm2y_diagnostic(
        tmp_path / "suite.json",
        out_dir=tmp_path / "diagnostic",
    )

    assert cfg.suite_result_json == str(tmp_path / "suite.json")
    assert cfg.gate_shapes == ("high", "equal", "low")
    assert cfg.rho_grid == (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    assert cfg.axis_modes == ("n_only", "v_only", "n_plus_v")
    assert cfg.eval_test is cfg.eval_holdout is False
    for changes in (
        {"eval_test": True},
        {"eval_holdout": True},
        {"rho_grid": (0.2,)},
        {"gate_shapes": ("high",)},
    ):
        with pytest.raises(ValueError):
            diagnostic.validate_hm2y_diagnostic_config(
                dataclasses.replace(cfg, **changes)
            )


def _decision_rows():
    accuracy = {
        f"{metric}@{k}": 1.0
        for metric in ("recall", "ndcg")
        for k in (10, 20, 50)
    }
    rows = [{"model_id": "m1", "gate_shape": "none", "rho": 0.0,
             "revenue@10": 1.0, **accuracy}]
    revenues = {
        0.05: (1.00, 1.00, 1.00),
        0.10: (1.04, 1.02, 1.03),
        0.15: (1.05, 1.03, 1.04),
        0.20: (1.06, 1.07, 1.04),
        0.30: (1.04, 1.03, 1.05),
        0.40: (0.99, 0.98, 0.98),
    }
    for rho, values in revenues.items():
        for model_id, revenue in zip(
            ("dual_clv_fixed", "dual_shuffled_user", "dual_adapter_only"),
            values,
            strict=True,
        ):
            rows.append(
                {
                    "model_id": model_id,
                    "gate_shape": "high",
                    "rho": rho,
                    "revenue@10": revenue,
                    **accuracy,
                }
            )
    return pd.DataFrame(rows)


def test_decision_requires_two_adjacent_joint_passes_and_selects_lowest_rho():
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    decision, table = diagnostic.diagnostic_decision(_decision_rows())

    assert decision["success"] is True
    assert decision["selected_gate_shape"] == "high"
    assert decision["selected_rho"] == pytest.approx(0.10)
    assert decision["plateau_rhos"] == [0.10, 0.15]
    by_rho = table.set_index("rho")
    assert bool(by_rho.loc[0.10, "joint_pass"]) is True
    assert bool(by_rho.loc[0.20, "joint_pass"]) is False


def test_decision_rejects_an_isolated_winning_point():
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    rows = _decision_rows()
    rows.loc[
        rows.model_id.eq("dual_adapter_only") & rows.rho.eq(0.15),
        "revenue@10",
    ] = 1.06

    decision, _ = diagnostic.diagnostic_decision(rows)

    assert decision["success"] is False
    assert decision["selected_rho"] is None
    assert decision["axis_diagnostic_gate_shape"] == "high"
    assert decision["axis_diagnostic_rho"] == pytest.approx(0.20)


def test_suite_payload_is_full_period_validation_only_and_hash_verified(tmp_path):
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    result, expected = _write_suite_payload(tmp_path)
    cfg = diagnostic.configure_hm2y_diagnostic(result)

    payload, paths = diagnostic.load_verified_suite_payload(cfg)

    assert payload["result_fingerprint"] == "fp"
    assert set(paths) == set(expected["checkpoint_paths"])
    assert all(path.is_file() for path in paths.values())

    paths["dual_clv_fixed"].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash"):
        diagnostic.load_verified_suite_payload(cfg)


def test_evaluate_model_grid_normalizes_each_axis_and_keeps_user_metrics():
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    class FakeModel:
        def __init__(self):
            self.gate = None
            self.axis = None

        def set_gate_shape(self, value):
            self.gate = value

        def set_eval_axes(self, value):
            self.axis = value

        def axis_diagnostics(self, gate_shape):
            assert gate_shape == self.gate
            return {
                "effective_n_ratio": 0.25,
                "effective_v_ratio": 0.50,
                "effective_total_ratio": 1.00,
            }

    calls = []

    def evaluator(model, lam, *_args, **_kwargs):
        calls.append((model.gate, model.axis, lam))
        return {"revenue@10": lam}, {"revenue": [lam]}

    prepared = {
        "cache": object(),
        "meta": object(),
        "data": object(),
        "base_cfg": object(),
    }
    rows, per_user = diagnostic.evaluate_model_grid(
        FakeModel(),
        "dual_clv_fixed",
        prepared,
        gate_shapes=("high",),
        rho_grid=(0.20,),
        axis_mode="v_only",
        evaluator=evaluator,
    )

    assert calls == [("high", "v_only", pytest.approx(0.40))]
    assert rows == [
        {
            "model_id": "dual_clv_fixed",
            "gate_shape": "high",
            "axis_mode": "v_only",
            "rho": 0.20,
            "raw_effective_ratio": 0.50,
            "lambda_equivalent": pytest.approx(0.40),
            "effective_strength": 0.20,
            "effective_n_ratio": 0.25,
            "effective_v_ratio": 0.50,
            "effective_total_ratio": 1.00,
            "revenue@10": pytest.approx(0.40),
        }
    ]
    assert per_user[("high", 0.20)]["revenue"] == [pytest.approx(0.40)]


def test_persist_writes_curve_decision_axis_and_provenance(tmp_path):
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    cfg = diagnostic.configure_hm2y_diagnostic(
        tmp_path / "suite.json",
        out_dir=tmp_path / "out",
    )
    curve = pd.DataFrame([
        {"model_id": "m1", "gate_shape": "none", "rho": 0.0,
         "revenue@10": 1.0}
    ])
    decision_table = pd.DataFrame([
        {"gate_shape": "high", "rho": 0.1, "joint_pass": True}
    ])
    axis = pd.DataFrame([
        {"model_id": "dual_clv_fixed", "axis_mode": "n_only",
         "gate_shape": "high", "rho": 0.1, "revenue@10": 1.1}
    ])
    payload = {
        "result_fingerprint": "source-fp",
        "checkpoint_paths": {"m1": "/m1.pt"},
        "checkpoint_sha256": {"m1": "abc"},
    }
    decision = {
        "success": True,
        "selected_gate_shape": "high",
        "selected_rho": 0.1,
    }

    result = diagnostic.persist_diagnostic(
        cfg, payload, curve, decision_table, axis, decision
    )

    assert set(result.attrs["result_paths"]) == {
        "curve_csv", "decision_csv", "axis_csv", "json"
    }
    assert all(Path(path).is_file() for path in result.attrs["result_paths"].values())
    saved = json.loads(Path(result.attrs["result_paths"]["json"]).read_text())
    assert saved["source_suite_result_fingerprint"] == "source-fp"
    assert saved["decision"] == decision
    assert saved["interpretation"]["training_executed"] is False
    assert saved["interpretation"]["test_executed"] is False


def test_public_runner_evaluates_full_curve_then_three_axis_modes(monkeypatch, tmp_path):
    import lightgcn_clv_dual_hm2y_diagnostic as diagnostic

    cfg = diagnostic.configure_hm2y_diagnostic(
        tmp_path / "suite.json", out_dir=tmp_path / "out"
    )
    payload = {"result_fingerprint": "source-fp"}
    monkeypatch.setattr(
        diagnostic,
        "load_verified_suite_payload",
        lambda _cfg: (payload, {name: tmp_path / f"{name}.pt" for name in (
            "m1", "encoder", "dual_clv_fixed", "dual_shuffled_user",
            "dual_adapter_only",
        )}),
    )
    accuracy = {
        f"{metric}@{k}": 1.0
        for metric in ("recall", "ndcg")
        for k in (10, 20, 50)
    }
    prepared = {
        "baseline_flat": {"revenue@10": 1.0, **accuracy},
        "cache": object(), "meta": object(), "data": object(),
        "base_cfg": object(),
    }
    monkeypatch.setattr(
        diagnostic, "_prepare_from_suite", lambda *_args: (object(), prepared)
    )
    monkeypatch.setattr(
        diagnostic, "_load_checkpoint_model", lambda model_id, *_args: model_id
    )

    revenues = {
        "dual_clv_fixed": 1.05,
        "dual_shuffled_user": 1.03,
        "dual_adapter_only": 1.04,
    }

    def fake_grid(_model, model_id, _prepared, *, gate_shapes, rho_grid,
                  axis_mode, evaluator=None, progress=False):
        rows = []
        for gate in gate_shapes:
            for rho in rho_grid:
                revenue = revenues[model_id]
                if axis_mode == "n_plus_v" and gate != "high":
                    revenue = 0.99
                rows.append({
                    "model_id": model_id,
                    "gate_shape": gate,
                    "axis_mode": axis_mode,
                    "rho": rho,
                    "revenue@10": revenue,
                    **accuracy,
                })
        return rows, {}

    monkeypatch.setattr(diagnostic, "evaluate_model_grid", fake_grid)

    result = diagnostic.run_hm2y_diagnostic(cfg)

    assert len(result) == 55  # M1 + 3 models × 3 gates × 6 rho
    assert result.attrs["decision"]["success"] is True
    assert result.attrs["decision"]["selected_gate_shape"] == "high"
    assert result.attrs["decision"]["selected_rho"] == pytest.approx(0.05)
    assert len(result.attrs["axis_rows"]) == 10  # M1 + 3 models × 3 axes


def test_colab_is_pinned_and_runs_evaluation_only_from_the_suite_json():
    notebook = json.loads(Path("clv_dual_hm2y_diagnostic_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", source)
    assert "clv_dual_hm2y_suite_*.json" in source
    assert "configure_hm2y_diagnostic" in source
    assert "run_hm2y_diagnostic(cfg)" in source
    assert "axis_csv" in source and "decision_csv" in source
    for forbidden in (
        "eval_test=True",
        "eval_holdout=True",
        "run_hm2y_suite",
        "_train_variant",
        "train_moe",
        "ACKNOWLEDGE_HIGH_COST",
    ):
        assert forbidden not in source
