from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import clv_m4_user_centered_final10 as s


def reference_fixture(tmp_path):
    root = tmp_path / "reference" / "run" / "reports"
    root.mkdir(parents=True)
    cfg = s.prior.configure()
    protocol = s.prior.preflight_summary(cfg) | {
        "config": json.loads(json.dumps(asdict(cfg))),
        "source_revision": s.REFERENCE_REVISION,
    }
    rows = [{"model_id": model, "seed": seed,
             **{m: 1.0 for m in (*s.prior.ACCURACY, *s.prior.ECONOMIC)},
             "user_value_tendency_recommended_price_alignment": .1}
            for seed in cfg.seeds for model in s.prior.models_for(cfg)]
    pd.DataFrame(rows).to_csv(root / "absolute.csv", index=False)
    pd.DataFrame([{"model_id": r["model_id"], "seed": r["seed"]} for r in rows]).to_csv(
        root / "diagnostics.csv", index=False)
    path = root / "result.json"
    path.write_text(json.dumps({"protocol": protocol, "reading": {"complete": True}}))
    return s.configure(reference_dir=str(root.parents[1]), out_dir=str(tmp_path / "new")), path


def test_reuse_requires_exact_final_split_config_and_all_ten_seeds(tmp_path):
    cfg, path = reference_fixture(tmp_path)
    refs = s.load_references(cfg)
    assert len(refs["absolute"]) == 30
    assert set(refs["absolute"].model_id) == set(s.REFERENCES)
    old = json.loads(path.read_text())
    old["protocol"]["split"] = "historical_development_days_684_690"
    path.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match="분할"):
        s.load_references(cfg)


def test_reference_missing_stops_before_preparation(tmp_path, monkeypatch):
    cfg = s.configure(reference_dir=str(tmp_path / "missing"), out_dir=str(tmp_path / "new"))
    monkeypatch.setattr(s.prior, "prepare", lambda *_: pytest.fail("must not prepare or train"))
    with pytest.raises(RuntimeError, match="완료 참조"):
        s.prepare(cfg)


def test_frozen_implementation_matches_original_run():
    assert set(s._verify_frozen_code()) == set(s.FROZEN_FILES)


def test_completed_m4_skips_model_construction_and_test_evaluation(tmp_path, monkeypatch):
    cfg, _ = reference_fixture(tmp_path)
    prepared = {"run_dir": tmp_path / "new", "references": s.load_references(cfg),
                "config_hash": "same", "revision": "rev", "input_hash": "input",
                "observed_weight_audit": {"sha256": "weights"}}
    row = {"identity": asdict(s._identity(prepared, 42)), "model_id": s.M4_C,
           "seed": 42, "epochs": 300,
           "metrics": {m: 1.0 for m in prepared["references"]["metrics"]},
           "diagnostics": {"graph": "binary", "row_weighted": True,
                           "weight_audit": {"sha256": "weights"}}}
    target = prepared["run_dir"] / "arms" / f"{s.M4_C}_s42.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(row))
    monkeypatch.setattr(s.prior, "_build_model", lambda *_: pytest.fail("no new fit"))
    monkeypatch.setattr(s.prior.fixed, "_evaluate", lambda *_: pytest.fail("no repeated test"))
    assert s.run_arm(prepared, 42) == row
    plan = s.execution_plan(prepared)
    assert plan.iloc[0].action == "reuse_completed"
    assert plan.iloc[0].max_remaining_epochs == 0
    assert plan.iloc[1:].action.eq("train_missing").all()
    row["diagnostics"]["weight_audit"]["sha256"] = "changed"
    target.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="신원"):
        s.run_arm(prepared, 42)


def test_binary_m4_joint_training_only_and_one_final_evaluation(tmp_path, monkeypatch):
    cfg, _ = reference_fixture(tmp_path)
    prepared = {"run_dir": tmp_path / "new", "references": s.load_references(cfg),
                "config_hash": "same", "revision": "rev", "input_hash": "input",
                "observed_weight_audit": {"sha256": "weights"},
                "observed_weights": np.array([.8, 1.2]),
                "training_cfg": s.prior.configure(), "data": {"adj": "binary"}}
    model = object()
    def build(arm, cfg, seed):
        assert arm["adjacency"] == "binary" and seed == 42
        return model
    def train(actual_model, actual_prepared, cfg, model_id, seed, store, row_weights):
        assert actual_model is model and actual_prepared is prepared
        assert model_id == s.M4_C and cfg.epochs == 300
        np.testing.assert_equal(row_weights, prepared["observed_weights"])
        return [{"epoch": ep} for ep in range(1, 301)]
    calls = []
    def evaluate(*_):
        calls.append(1)
        return {m: 1.0 for m in prepared["references"]["metrics"]}
    monkeypatch.setattr(s.prior, "_build_model", build)
    monkeypatch.setattr(s.prior.fixed, "_train", train)
    monkeypatch.setattr(s.prior.fixed, "_evaluate", evaluate)
    s.run_arm(prepared, 42)
    s.run_arm(prepared, 42)
    assert len(calls) == 1


def test_full_report_factorial_and_partial_report_block(tmp_path):
    cfg, _ = reference_fixture(tmp_path)
    refs = s.load_references(cfg)
    refs["absolute"].loc[refs["absolute"].model_id.eq(s.prior.M3), "recall@10"] = 1.1
    refs["absolute"].loc[refs["absolute"].model_id.eq(s.prior.M5_C), "recall@10"] = 1.3
    prepared = {"run_dir": tmp_path / "new", "references": refs, "protocol": {}}
    rows = [{"model_id": s.M4_C, "seed": seed,
             "metrics": {m: 1.05 for m in refs["metrics"]}, "diagnostics": {"graph": "binary"}}
            for seed in s.prior.SEEDS]
    result = s.report(prepared, rows)
    assert result["summary"]["count"].eq(10).all()
    assert len(result["absolute"]) == 40
    assert "user_value_tendency_recommended_price_alignment" in set(result["summary"].metric)
    np.testing.assert_allclose(result["factorial"].loc[
        result["factorial"].metric.eq("recall@10"), "interaction"], .15)
    assert all(Path(p).is_file() for p in result["paths"].values())
    assert result["reading"]["clv_attribution_claim"] is False
    with pytest.raises(RuntimeError, match="각10시드"):
        s.report(prepared, rows[:-1])
