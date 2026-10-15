from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import clv_m5_ab_adaptive_lastweek_seed49 as s
from test_clv_dataset_adaptive_orthogonal_nv import _features


@pytest.fixture
def prepared(tmp_path):
    features = _features()
    users = np.array([0, 0, 1, 1, 2, 2])
    items = np.array([0, 1, 1, 2, 2, 3])
    binary = s.v3.build_adj(users, items, np.ones(6, np.float32), 3, 5)
    weighted = s.v3.build_adj(users, items, np.array([.8, 1.2, 1.1, .9, .7, 1.3], np.float32), 3, 5)
    metrics = [f"metric_{i}" for i in range(142)]
    refs = pd.DataFrame([{"model_id": m, "seed": 49, **dict.fromkeys(metrics, 1.)}
                         for m in s.LABELS if m not in s.MODELS])
    return {"cfg": s.configure(out_dir=str(tmp_path)), "run_dir": tmp_path,
            "revision": "test", "config_hash": "test", "input_hash": "test",
            "features": features, "feature_sha256": s.m2._feature_hash(features),
            "observed_adj": weighted, "observed_weights": np.array([.8, 1.2, 1.1, .9, .7, 1.3]),
            "data": {"n_users": 3, "n_items": 5, "adj": binary},
            "protocol": {}, "references": {"absolute": refs, "metrics": metrics,
                "diagnostics": refs[["model_id", "seed"]].copy(), "audit": {}}}


def test_fixed_seed_and_only_two_arms(prepared):
    s._validate_config(prepared["cfg"])
    for change in ({"seeds": (53,)}, {"epochs": 100}, {"input_days": 704}, {"bandwidth": .5}):
        with pytest.raises(ValueError):
            s._validate_config(replace(prepared["cfg"], **change))
    assert s.arm_inputs(prepared, s.M5_A)[0] is prepared["observed_adj"]
    assert s.arm_inputs(prepared, s.M5_A)[1] is None
    assert s.arm_inputs(prepared, s.M5_B)[0] is prepared["data"]["adj"]
    assert s.arm_inputs(prepared, s.M5_B)[1] is prepared["observed_weights"]
    with pytest.raises(ValueError):
        s.arm_inputs(prepared, s.prior.M1)


@pytest.mark.parametrize("model_id", s.MODELS)
def test_real_forward_joint_gradient_for_each_combination(prepared, model_id):
    adj, weights = s.arm_inputs(prepared, model_id)
    model = s.m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=3, n_items=5, features=prepared["features"], adj=adj,
        id_dim=8, n_layers=2, pref_reg=.001, shared_l2=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([[4], [4], [0]])
    w = None if weights is None else torch.tensor(weights[:3], dtype=torch.float32)
    loss, _, _ = s.fixed.recheck._batch_loss(model, u, pos, neg, w)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.axis_logits.grad is not None and model.axis_logits.grad.norm() > 0
    assert model.rho_logit.grad is not None and model.rho_logit.grad.abs() > 0
    assert all(layer.weight.grad is not None for layer in model.encoders.values())
    optimizer.step()


def test_two_fits_only_once_and_reports_partial(prepared, monkeypatch):
    trained, evaluated = [], []
    def train(model, p, cfg, model_id, seed, store, row_weights=None):
        assert seed == 49 and cfg.epochs == 300
        trained.append((model_id, row_weights is not None))
        return [{"epoch": i} for i in range(1, 301)]
    def evaluate(model, p):
        evaluated.append(model)
        return dict.fromkeys(p["references"]["metrics"], 1.01)
    monkeypatch.setattr(s.fixed, "_train", train)
    monkeypatch.setattr(s.fixed, "_evaluate", evaluate)
    before = s.report(prepared)
    assert not before["complete"] and len(before["absolute"]) == 5
    s.run_arm(prepared, s.M5_A)
    middle = s.report(prepared)
    assert middle["missing"] == ["M5_B"] and len(middle["absolute"]) == 6
    result = s.run(prepared)
    assert result["complete"] and len(result["absolute"]) == 7
    assert len(result["comparison"]) == 2 * 5 * 142
    assert set(result["absolute"].seed) == {49}
    s.run(prepared)
    assert trained == [(s.M5_A, False), (s.M5_B, True)] and len(evaluated) == 2


def test_incomplete_training_never_evaluates(prepared, monkeypatch):
    monkeypatch.setattr(s.fixed, "_train", lambda *a, **kw: [{"epoch": 299}])
    monkeypatch.setattr(s.fixed, "_evaluate", lambda *a: pytest.fail("test evaluated too soon"))
    with pytest.raises(RuntimeError, match="300epoch"):
        s.run_arm(prepared, s.M5_A)


def test_reference_mismatch_stops_before_prepare(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text("{}")
    (tmp_path / "absolute.csv").write_text("seed\n49\n")
    (tmp_path / "diagnostics.csv").write_text("seed\n49\n")
    monkeypatch.setattr(s.prior, "prepare", lambda *a: pytest.fail("data preparation started"))
    with pytest.raises(RuntimeError, match="原本|원본"):
        s.prepare(s.configure(m2_reference=str(tmp_path)))


def test_completed_identity_mismatch_rejected(prepared):
    path = prepared["run_dir"] / "arms" / f"{s.M5_A}.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"identity": {}, "epochs": 300, "metrics": {}}))
    with pytest.raises(RuntimeError, match="신원"):
        s.run_arm(prepared, s.M5_A)


@pytest.mark.parametrize("dataset,seed,count", [("hm", 49, 1), ("dunnhumby", 50, 2)])
def test_new_batches_fit_only_requested_models(prepared, monkeypatch, dataset, seed, count):
    cfg = s.configure(dataset=dataset, seed=seed, out_dir=str(prepared["run_dir"]))
    prepared["cfg"] = cfg
    labels, models = s.labels_for(cfg), s.models_for(cfg)
    refs = pd.DataFrame([{"model_id": m, "seed": seed,
                        **dict.fromkeys(prepared["references"]["metrics"], 1.)}
                        for m in labels if m not in models])
    prepared["references"].update(absolute=refs, diagnostics=refs[["model_id", "seed"]])
    if dataset == "hm":
        del prepared["observed_adj"]  # HM must not require or build an M3 graph.
    trained, evaluated = [], []
    def train(model, p, config, model_id, run_seed, store, row_weights=None):
        assert config == cfg and run_seed == seed
        assert (row_weights is not None) == (labels[model_id] == "M5_B")
        trained.append(model_id)
        return [{"epoch": i} for i in range(1, 301)]
    def evaluate(model, p):
        evaluated.append(True)
        return dict.fromkeys(p["references"]["metrics"], 1.01)
    monkeypatch.setattr(s.fixed, "_train", train)
    monkeypatch.setattr(s.fixed, "_evaluate", evaluate)
    result = s.run(prepared)
    s.run(prepared)
    assert trained == list(models) and len(evaluated) == count
    assert result["complete"] and result["reading"]["new_fit_count"] == count
    assert result["reading"]["reused_count"] == (4 if dataset == "hm" else 5)
    assert set(result["absolute"].seed) == {seed}
    assert result["absolute"].model.notna().all()
    assert len(result["comparison"]) == count * len(refs) * 142
    s.io._atomic_json(prepared["run_dir"] / "protocol.json", prepared["protocol"])
    s.io._atomic_json(prepared["run_dir"] / "reference_audit.json", prepared["references"]["audit"])
    assert s.export_zip(prepared, prepared["run_dir"] / "result.zip")


def test_hm_input_path_uses_binary_m2_and_m4_only(monkeypatch):
    data = {"train": object(), "n_users": 2, "n_items": 3}
    prepared = {"data": data, "axes": object(), "features": object()}
    monkeypatch.setattr(s.m2, "prepare", lambda cfg: prepared)
    monkeypatch.setattr(s.hm, "validate_split", lambda data: None)
    monkeypatch.setattr(s.fixed.evaluation, "build_clv_inputs", lambda axes:
        (np.array([.2, .7]), np.array([.3, .8]), np.array([.1, .9]), np.array([True, False])))
    def economics(train, **kw):
        assert kw["q_n"][1] == 0 and kw["q_c"][1] == 0
        return dict.fromkeys(("item_amount_percentile", "item_economic_valid",
                             "user_economic_valid", "user_bin_fit", "item_bin"), None)
    monkeypatch.setattr(s.hm.economics, "build_nv_economic_inputs", economics)
    monkeypatch.setattr(s.hm.m4c, "row_weights", lambda p: (np.ones(3), {"sha256": "test"}))
    monkeypatch.setattr(s.prior, "prepare", lambda *a: pytest.fail("HM builds M3 unnecessarily"))
    result = s._prepare_hm_inputs(s.configure(dataset="hm"))
    assert "observed_adj" not in result and result["observed_weight_audit"]["sha256"] == "test"
