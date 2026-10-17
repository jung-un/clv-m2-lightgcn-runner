from dataclasses import replace
import hashlib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

import clv_hm_m5_d_adaptive_lastweek_seed49 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared


@pytest.fixture
def hm(prepared):
    prepared["cfg"] = s.configure(out_dir=str(prepared["run_dir"]))
    prepared["observed_weights"] = np.asarray(prepared["observed_weights"], np.float32)
    audit = {"sha256": hashlib.sha256(prepared["observed_weights"].tobytes()).hexdigest()}
    prepared.update(observed_weight_audit=audit, protocol={"m4_weight_audit": audit})
    refs = pd.DataFrame([{"model_id": mid, "seed": 49,
                        **dict.fromkeys(prepared["references"]["metrics"], 1.)}
                        for mid in s.LABELS if mid != s.MODEL_ID])
    prepared["references"].update(absolute=refs, diagnostics=refs[["model_id", "seed"]])
    for name in ("protocol.json", "reference_audit.json"):
        s.base.io._atomic_json(prepared["run_dir"] / name, {})
    return prepared


def test_one_fit_weighted_bpr_and_cached_rerun(hm, monkeypatch):
    fits, evaluations = [], []
    original = s.base.m2.DatasetAdaptiveOrthogonalNVLightGCN
    def build(**kwargs):
        assert kwargs["adj"] is hm["observed_adj"]
        return original(**kwargs)
    monkeypatch.setattr(s.base.m2, "DatasetAdaptiveOrthogonalNVLightGCN", build)
    def train(model, p, cfg, mid, seed, store, row_weights=None):
        assert row_weights is hm["observed_weights"] and mid == s.MODEL_ID and seed == 49
        assert cfg.dataset == "hm" and cfg.batch_size == 131072 and cfg.epochs == 300
        fits.append(mid)
        return [{"epoch": i} for i in range(1, 301)]
    def evaluate(model, p):
        evaluations.append(True)
        return dict.fromkeys(p["references"]["metrics"], 1.01)
    monkeypatch.setattr(s.base.fixed, "_train", train)
    monkeypatch.setattr(s.base.fixed, "_evaluate", evaluate)
    assert not s.report(hm)["complete"]
    with pytest.raises(RuntimeError, match="완료"):
        s.export_zip(hm, hm["run_dir"] / "early.zip")
    result = s.run(hm)
    assert result["complete"] and len(result["absolute"]) == 5 and len(result["comparison"]) == 4*142
    assert set(result["comparison"].reference) == {"M1", "M2", "M3", "M4"}
    assert np.allclose(result["comparison"].relative_change_pct, 1.)
    assert s.completed(hm)["diagnostics"]["row_weighted"] is True
    s.run(hm)
    assert fits == [s.MODEL_ID] and len(evaluations) == 1
    with ZipFile(s.export_zip(hm, hm["run_dir"] / "done.zip")) as z:
        assert len([n for n in z.namelist() if n.startswith("arms/")]) == 1


def test_m4_and_early_test_blocked(hm, monkeypatch):
    monkeypatch.setattr(s.base.fixed, "_train", lambda *a, **kw: [{"epoch": 299}])
    monkeypatch.setattr(s.base.fixed, "_evaluate", lambda *a: pytest.fail("early test"))
    with pytest.raises(RuntimeError, match="300epoch"):
        s.run(hm)
    hm["observed_weights"] = None
    with pytest.raises(RuntimeError, match="M4"):
        s.run(hm)


def test_frozen_config_and_identity(hm):
    for change in ({"epochs": 100}, {"seeds": (50,)}, {"input_days": 704}, {"batch_size": 8192}):
        with pytest.raises(ValueError):
            s.validate_config(replace(hm["cfg"], **change))
    s.base.io._atomic_json(hm["run_dir"] / "arms" / f"{s.MODEL_ID}.json", {"identity": {}})
    with pytest.raises(RuntimeError, match="신원"):
        s.run(hm)


def test_joint_gradient_with_weighted_graph_and_weighted_loss(hm):
    model = s.base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=3, n_items=5, features=hm["features"], adj=hm["observed_adj"],
        id_dim=8, n_layers=2, pref_reg=.001, shared_l2=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([[4], [4], [0]])
    loss, _, _ = s.base.fixed.recheck._batch_loss(model, u, pos, neg, torch.tensor([.8, 1., 1.2]))
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.axis_logits.grad.norm() > 0 and model.rho_logit.grad.abs() > 0
    assert all(layer.weight.grad is not None for layer in model.encoders.values())
    optimizer.step()


@pytest.mark.parametrize("bad_graph", [False, True])
@pytest.mark.parametrize("bad_weights", [False, True])
def test_prepare_no_training_with_m4_and_graph_mismatch_gate(hm, monkeypatch, bad_graph, bad_weights):
    cfg = hm["cfg"]
    data = hm["data"]
    data.update(train=pd.DataFrame(), data_stats={"test": "lastweek"}, pos_key=hm["features"]["keys"])
    gc = s.base.hm.configure()
    audit = {"coefficient_of_variation": .2}
    p2 = {"input_hash": hm["input_hash"], "data_stats": data["data_stats"], "feature_sha256": hm["feature_sha256"]}
    p3 = {"config": {k: getattr(gc, k) for k in ("target_cv", "max_degree_correlation", "max_price_correlation", "max_popularity_correlation")},
          "m3_audit": audit, "split": "lastweek", "fixed": {}, "m4_c_weight_audit": hm["observed_weight_audit"]}
    if bad_weights:
        p3["m4_c_weight_audit"] = {"sha256": "changed"}
    hm["references"]["sources"] = {"m2": {"protocol": p2}, "m4": {"protocol": p3}}
    hm.update(axes={}, protocol={"dataset_initialization": {}})
    monkeypatch.setattr(s.base, "load_references", lambda cfg: hm["references"])
    hm["item_amount_percentile"] = np.ones(5)
    monkeypatch.setattr(s.base, "_prepare_hm_inputs", lambda cfg: hm)
    monkeypatch.setattr(s.base.hm, "validate_split", lambda data: None)
    monkeypatch.setattr(s.base.fixed.evaluation, "build_clv_inputs", lambda axes:
        (np.array([.2, .5, .8]), np.array([.3, .6, .9]), np.ones(3), np.ones(3, bool)))
    monkeypatch.setattr(s.base.hm.hm_graph, "_purchase_keyed", lambda train: train)
    monkeypatch.setattr(s.base.fixed.graph, "centered_edge_signals", lambda *a: {
        "edge_users": data["pos_key"] // 5, "edge_items": data["pos_key"] % 5})
    monkeypatch.setattr(s.base.fixed, "build_graph", lambda *a: {
        "audit": {"coefficient_of_variation": .3 if bad_graph else .2}, "beta": .1,
        "weights": np.ones(len(data["pos_key"]))})
    monkeypatch.setattr(s.base.hm.m4c, "row_weights", lambda *a: pytest.fail("M4 weights built"))
    monkeypatch.setattr(s.base.fixed, "_train", lambda *a, **kw: pytest.fail("fit in prepare"))
    if bad_weights:
        with pytest.raises(RuntimeError, match="손실 가중치"):
            s.prepare(cfg)
    elif bad_graph:
        with pytest.raises(RuntimeError, match="그래프 진단"):
            s.prepare(cfg)
    else:
        result = s.prepare(cfg)
        assert result["protocol"]["new_fit_models"] == [s.MODEL_ID]
        assert result["protocol"]["m4_intervention"] is True
        assert len(s.report(result)["absolute"]) == 4


def test_bad_reference_stops_before_expensive_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base.m2, "prepare", lambda *a: pytest.fail("data preparation started"))
    with pytest.raises(RuntimeError, match="원본"):
        s.prepare(s.configure(m2_reference=str(tmp_path)))

