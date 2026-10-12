from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

import clv_m5_adaptive_nv_m3_m4c_seed48 as s
from test_clv_dataset_adaptive_orthogonal_nv import _features


def _reference(tmp_path):
    cfg = s.configure(out_dir=str(tmp_path / "new"), reference_json=str(tmp_path / "old/reports/result.json"))
    arms, rows = [], []
    for model, factor in ((s.M1, 1.0), (s.REFERENCE, 1.1)):
        curve = [{"epoch": ep, "loss": .5, "p_correct": .9,
                  "metrics": {m: factor for m in (*s.adaptive.ACCURACY, *s.adaptive.ECONOMIC,
                             "user_value_tendency_recommended_price_alignment")}}
                 for ep in range(25, 301, 25)]
        arms.append({"model_id": model, "seed": 48, "curve": curve,
                     "identity": asdict(s.RunIdentity(s.REFERENCE_VERSION, model, 48,
                                                       "old", s.REFERENCE_REVISION, "input")),
                     "code_version": s.adaptive.REFERENCE_CODE_VERSION,
                     "source_revision": s.adaptive.REFERENCE_REVISION, "input_hash": "input",
                     "m3_weight_sha256": "graph", "m4_weight_sha256": "weights"})
        rows.extend({"model_id": model, "seed": 48, "epoch": r["epoch"], **r["metrics"]} for r in curve)
    old = {"code_version": s.REFERENCE_VERSION, "source_revision": s.REFERENCE_REVISION,
           "input_hash": "input", "config_hash": "old", "config": asdict(cfg) | {"seeds": [48]},
           "split": s.SPLIT, "final_test": False, "holdout": False,
           "train_days": [1, 683], "development_days": [684, 690], "arms": arms,
           "m3_weight_sha256": "graph", "m4_audit": {"sha256": "weights"}}
    path = tmp_path / "old/reports/result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old))
    pd.DataFrame(rows).to_csv(path.parent / "absolute.csv", index=False)
    return cfg, s.load_reference(cfg)


def test_fixed_seed48_protocol_and_reference_identity(tmp_path):
    cfg, refs = _reference(tmp_path)
    assert len(refs["absolute"]) == 24
    with pytest.raises(ValueError):
        s.configure(seeds=(43,))
    with pytest.raises(ValueError):
        s.validate_config(replace(cfg, epochs=100))
    path = tmp_path / "old/reports/result.json"
    old = json.loads(path.read_text())
    old["split"] = "lastweek_test"
    path.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match="개발분할"):
        s.load_reference(cfg)


def test_missing_reference_blocks_preparation_and_training(tmp_path, monkeypatch):
    cfg = s.configure(reference_json=str(tmp_path / "missing.json"), out_dir=str(tmp_path / "new"))
    monkeypatch.setattr(s.m3, "_prepare", lambda *_: pytest.fail("should fail before data prep"))
    with pytest.raises(FileNotFoundError):
        s.prepare(cfg)


def test_existing_m2_m3_m4_code_is_reused_unchanged():
    assert len(s._verify_source()) == 5


@pytest.mark.parametrize("changed", ["input", "graph", "weight"])
def test_mismatched_input_or_intervention_stops_before_training(tmp_path, monkeypatch, changed):
    cfg, refs = _reference(tmp_path)
    graph = {"weights": np.array([.8, 1.2], np.float32), "beta": .2}
    weights = np.array([.9, 1.1], np.float32)
    old = refs["payload"]
    old["m3_beta"] = graph["beta"]
    old["m3_weight_sha256"] = s.hashlib.sha256(graph["weights"].tobytes()).hexdigest()
    old["m4_audit"]["sha256"] = s.hashlib.sha256(weights.tobytes()).hexdigest()
    old[{"input": "input_hash", "graph": "m3_weight_sha256"}.get(changed, "input_hash")] = (
        "changed" if changed != "weight" else old["input_hash"])
    if changed == "weight":
        old["m4_audit"]["sha256"] = "changed"
    monkeypatch.setattr(s, "load_reference", lambda *_: refs)
    monkeypatch.setattr(s, "_verify_source", lambda: {})
    monkeypatch.setattr(s.m3, "_prepare", lambda *_: {"input_hash": "input"})
    monkeypatch.setattr(s, "validate_prepared", lambda *_: None)
    monkeypatch.setattr(s.m3, "build_arm_graph", lambda *_: graph)
    monkeypatch.setattr(s.m4c, "row_weights", lambda *_: (weights, {"sha256": s.hashlib.sha256(weights.tobytes()).hexdigest()}))
    monkeypatch.setattr(s, "build_model", lambda *_: pytest.fail("no new fit"))
    with pytest.raises(RuntimeError, match="입력·M3그래프·M4-C"):
        s.prepare(cfg)


def test_joint_model_uses_m3_graph_and_all_m2_bpr_gradients(tmp_path):
    cfg = s.configure(out_dir=str(tmp_path))
    features = _features()
    users = np.array([0, 0, 1, 1, 2, 2])
    items = np.array([0, 1, 1, 2, 2, 3])
    graph_weights = np.array([.8, 1.2, 1.1, .9, .7, 1.3], np.float32)
    prepared = {"data": {"n_users": 3, "n_items": 5}, "features": features,
                "signals": {"edge_users": users, "edge_items": items},
                "m3_graph": {"weights": graph_weights}}
    model = s.build_model(cfg, prepared)
    expected = s.v3.build_adj(users, items, graph_weights, 3, 5)
    torch.testing.assert_close(model.adj.to_dense(), expected.to_dense())
    u, pos, neg = [torch.tensor(x, device=s.v3.DEVICE) for x in ([0, 1, 2], [1, 2, 3], [4, 4, 0])]
    loss, _, _ = s.m3.recheck._batch_loss(model, u, pos, neg[:, None],
                                        torch.tensor([.8, 1.1, 1.2], device=s.v3.DEVICE))
    loss.backward()
    for parameter in (model.E_u.weight, model.E_i.weight, model.axis_logits, model.rho_logit,
                      *(layer.weight for layer in model.encoders.values())):
        assert parameter.requires_grad and parameter.grad is not None
        assert parameter.grad.norm() > 0
    assert model.axis_weights()[0] >= .15 and model.axis_weights()[1] >= .15
    assert all(p.requires_grad for p in model.parameters())


def test_completed_candidate_skips_training_and_scoring(tmp_path, monkeypatch):
    cfg, refs = _reference(tmp_path)
    prepared = {"out_dir": tmp_path / "new", "config_hash": "new", "revision": "rev",
                "input_hash": "input", "m3_graph": {"sha256": "graph"},
                "preflight": {"m4_audit": {"sha256": "weights"}}}
    arm = {"identity": asdict(s._identity(prepared)), "model_id": s.MODEL_ID, "seed": 48,
           "curve": [{"epoch": 300}], "m3_weight_sha256": "graph", "m4_weight_sha256": "weights"}
    path = prepared["out_dir"] / "arms/new" / f"{s.MODEL_ID}_s48.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(arm))
    monkeypatch.setattr(s, "build_model", lambda *_: pytest.fail("completed fit must not rebuild"))
    monkeypatch.setattr(s.capacity, "_train_curve", lambda *_: pytest.fail("no training"))
    assert s._run_arm(cfg, prepared) == arm


def test_report_compares_c_primary_preserves_full_metrics_and_fixed300(tmp_path):
    cfg, refs = _reference(tmp_path)
    prepared = {"out_dir": tmp_path / "new", "reference": refs, "config_hash": "new", "preflight": {}}
    metrics = refs["metrics"]
    curve = [{"epoch": ep, "loss": .2, "p_correct": .95,
              "metrics": {m: 1.12 if ep == 300 else 2.0 for m in metrics}}
             for ep in range(25, 301, 25)]
    result = s.report(cfg, prepared, {"curve": curve})
    assert result["reading"]["primary_reference"] == s.REFERENCE
    assert result["reading"]["exploratory_screen_rule_passed"]
    assert len(result["absolute"]) == 36
    assert set(result["comparison"].metric) == set(metrics)
    assert result["reading"]["clv_attribution_claim"] is False
    # Better intermediate points cannot replace the fixed final operating point.
    curve[-1]["metrics"]["recall@50"] = 1.05
    failed = s.report(cfg, prepared, {"curve": curve})
    assert not failed["reading"]["exploratory_screen_rule_passed"]
