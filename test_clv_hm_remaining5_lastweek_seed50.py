from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

import clv_hm_remaining5_lastweek_seed50 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared


@pytest.fixture
def hm(prepared, tmp_path):
    p = prepared
    p["cfg"] = s.configure(out_dir=str(tmp_path / "new"), core_reference=str(tmp_path / "core"))
    p["observed_weights"] = np.asarray(p["observed_weights"], np.float32)
    p["protocol"] = {"m4_weight_audit": {"sha256": hashlib.sha256(p["observed_weights"].tobytes()).hexdigest()}}
    p["data"]["pos_key"] = p["features"]["keys"]
    p.update(q_n=np.ones(3, np.float32)*.5, q_v=np.ones(3, np.float32)*.5,
             q_c=np.ones(3, np.float32)*.5, clv_valid=np.ones(3, bool),
             item_amount_percentile=np.ones(5, np.float32)*.5, item_economic_valid=np.ones(5, bool))
    core_dir = tmp_path / "core" / "abc"
    cp = {"source_revision": "old", "input_hash": "test"}
    s.base.io._atomic_json(core_dir / "protocol.json", cp)
    m1 = {"identity": asdict(s.base.RunIdentity(s.base.hm.CODE_VERSION, s.base.hm.M1, 50, "abc", "old", "test")),
          "model_id": s.base.hm.M1, "seed": 50, "epochs": 300,
          "metrics": dict.fromkeys(p["references"]["metrics"], 1.), "diagnostics": {}}
    s.base.io._atomic_json(core_dir / "arms" / f"{s.base.hm.M1}_s50.json", m1)
    p["references"].update(core_dir=core_dir, core_protocol=cp, m1=m1)
    for name in ("protocol.json", "reference_audit.json"):
        s.base.io._atomic_json(p["run_dir"] / name, {})
    return p


def test_five_routes_and_config(hm):
    assert [s.LABELS[mid] for mid in s.MODELS] == ["M2", "M4", "M5_A", "M5_B", "M5_D"]
    for mid in s.MODELS:
        adj, weights = s.arm_inputs(hm, mid)
        assert adj is (hm["observed_adj"] if mid in (s.M5_A, s.M5_D) else hm["data"]["adj"])
        assert (weights is not None) == (mid in (s.M4, s.M5_B, s.M5_D))
    for mid in s.base.hm.CORE_MODELS:
        with pytest.raises(ValueError):
            s.arm_inputs(hm, mid)
    for change in ({"seeds": (49,)}, {"epochs": 100}, {"input_days": 704}, {"batch_size": 8192}):
        with pytest.raises(ValueError):
            s.validate_config(replace(hm["cfg"], **change))
    with pytest.raises(ValueError, match="겹칠"):
        s.validate_config(replace(hm["cfg"], out_dir=hm["cfg"].core_reference + "/nested"))


def test_exact_five_fits_partial_reports_and_cached_rerun(hm, monkeypatch):
    fits, evaluations = [], []
    def train(model, p, cfg, mid, seed, store, row_weights=None):
        assert seed == 50 and cfg.epochs == 300 and cfg.batch_size == 131072
        assert isinstance(model, s.base.m2.DatasetAdaptiveOrthogonalNVLightGCN) == (mid != s.M4)
        assert row_weights is s.arm_inputs(p, mid)[1]
        fits.append(mid)
        return [{"epoch": i} for i in range(1, 301)]
    def evaluate(model, p):
        evaluations.append(True)
        return dict.fromkeys(p["references"]["metrics"], 1.01)
    monkeypatch.setattr(s.base.fixed, "_train", train)
    monkeypatch.setattr(s.base.fixed, "_evaluate", evaluate)
    assert not s.report(hm)["complete"]
    s.run_arm(hm, s.M2)
    assert len(s.report(hm)["absolute"]) == 2
    result = s.run(hm)
    assert result["complete"] and not result["reading"]["eight_models_complete"]
    assert len(result["absolute"]) == 6 and set(result["absolute"].seed) == {50}
    assert set(result["comparison"].reference) == {"M1", "M2", "M4", "M5_A", "M5_B", "M5_D"}
    s.run(hm)
    assert fits == list(s.MODELS) and len(evaluations) == 5
    with ZipFile(s.export_zip(hm, hm["run_dir"] / "done.zip")) as archive:
        assert len([p for p in archive.namelist() if p.startswith("arms/")]) == 5
        assert not any(p.endswith(".pt") for p in archive.namelist())


def test_optional_other_gpu_results_do_not_change_resume_identity(hm):
    before = asdict(s.identity(hm, s.M2))
    row = dict(hm["references"]["m1"])
    for mid in (s.base.hm.M3, s.base.hm.M5_C):
        row = row | {"model_id": mid, "identity": row["identity"] | {"model_id": mid}}
        s.base.io._atomic_json(hm["references"]["core_dir"] / "arms" / f"{mid}_s50.json", row)
    assert len(s.report(hm)["absolute"]) == 3
    assert asdict(s.identity(hm, s.M2)) == before


def test_early_evaluation_and_missing_weights_blocked(hm, monkeypatch):
    monkeypatch.setattr(s.base.fixed, "_train", lambda *a, **kw: [{"epoch": 299}])
    monkeypatch.setattr(s.base.fixed, "_evaluate", lambda *a: pytest.fail("early test"))
    with pytest.raises(RuntimeError, match="300epoch"):
        s.run_arm(hm, s.M2)
    hm["observed_weights"] = None
    with pytest.raises(RuntimeError, match="가중치 누락"):
        s.run_arm(hm, s.M4)


def test_wrong_seed_result_rejected(hm):
    m1 = hm["references"]["m1"] | {"seed": 49}
    s.base.io._atomic_json(hm["references"]["core_dir"] / "arms" / f"{s.base.hm.M1}_s50.json", m1)
    with pytest.raises(RuntimeError, match="신원"):
        s.report(hm)


def test_reference_preflight_requires_m1_before_preparation(hm, monkeypatch):
    cfg = hm["cfg"]
    cp = {"code_version": s.base.hm.CODE_VERSION, "config": asdict(cfg) | {
        k: getattr(s.base.hm.configure(), k) for k in s.GRAPH_KEYS},
        "source_revision": "old", "input_hash": "test", "data_stats": {"split": "lastweek"},
        "m3_audit": {"cv": .2}, "m4_c_weight_audit": {"sha256": "weight"}}
    refs = {"protocol": cp, "absolute": hm["references"]["absolute"], "audit": {}}
    monkeypatch.setattr(s.base, "_read_reference", lambda *a: refs)
    monkeypatch.setattr(s.base, "_prepare_hm_inputs", lambda *a: pytest.fail("expensive preparation"))
    protocol_path = hm["references"]["core_dir"] / "protocol.json"
    s.base.io._atomic_json(protocol_path, cp)
    assert s.load_references(cfg)["m1"]["seed"] == 50
    cfg = replace(cfg, core_reference=str(hm["run_dir"] / "empty"))
    with pytest.raises(RuntimeError, match="M1 seed50"):
        s.prepare(cfg)


def test_real_train_epoch_resume_same_model_and_optimizer(hm, monkeypatch, tmp_path):
    """Tiny synthetic data: shared training path must save each epoch and resume exactly."""
    torch.set_num_threads(1)
    hm["data"].update(tr_u=np.array([0, 0, 1, 1, 2, 2]),
                      tr_i=np.array([0, 1, 1, 2, 2, 3]))
    cfg = replace(hm["cfg"], epochs=3, batch_size=3)
    def build():
        s.base.v3.set_seed(50)
        return s.base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
            n_users=3, n_items=5, features=hm["features"], adj=hm["observed_adj"],
            id_dim=8, n_layers=2, pref_reg=.001, shared_l2=1e-6)
    identity = s.identity(hm, s.M5_D)
    full, resumed = build(), build()
    full_store = s.ProgressStore(tmp_path / "full", identity)
    split_store = s.ProgressStore(tmp_path / "split", identity)
    full_history = s.base.fixed._train(full, hm, cfg, s.M5_D, 50, full_store, row_weights=hm["observed_weights"])
    saved = split_store.save_epoch
    def interrupt(*args, **kwargs):
        saved(*args, **kwargs)
        if kwargs["epoch"] == 1:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(split_store, "save_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        s.base.fixed._train(resumed, hm, cfg, s.M5_D, 50, split_store, row_weights=hm["observed_weights"])
    monkeypatch.setattr(split_store, "save_epoch", saved)
    resumed = build()
    history = s.base.fixed._train(resumed, hm, cfg, s.M5_D, 50, split_store, row_weights=hm["observed_weights"])
    assert history == full_history
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_corrupt_checkpoint_preserved_no_silent_restart(hm):
    store = s.ProgressStore(hm["run_dir"] / "corrupt", s.identity(hm, s.M2))
    store.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    store.latest_checkpoint.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="자동 재학습하지"):
        store.restore_epoch(None, None, None)
    assert store.latest_checkpoint.read_bytes() == b"truncated"


@pytest.mark.parametrize("mismatch", [None, "input", "features", "weights", "graph"])
def test_prepare_shared_inputs_and_fingerprint_gates(hm, monkeypatch, mismatch):
    p = hm
    p["data"].update(train=None, data_stats={"test": "lastweek"})
    p["base_cfg"] = {}
    weight_audit = p["protocol"]["m4_weight_audit"]
    p.update(observed_weight_audit=weight_audit, protocol={"dataset_initialization": {"test": True}})
    p2 = {"input_hash": p["input_hash"], "data_stats": p["data"]["data_stats"],
          "feature_sha256": p["feature_sha256"], "split": "lastweek"}
    p3 = {"m4_c_weight_audit": weight_audit, "m3_audit": {"cv": .2}, "fixed": {}}
    if mismatch == "input": p2["input_hash"] = "other"
    if mismatch == "features": p2["feature_sha256"] = "other"
    if mismatch == "weights": p3["m4_c_weight_audit"] = {"sha256": "other"}
    refs = p["references"]
    refs.update(design={"m2": {"protocol": p2}, "m4": {"protocol": p3}}, audit={})
    calls = []
    monkeypatch.setattr(s, "load_references", lambda cfg: refs)
    def prepare_inputs(cfg):
        calls.append(True)
        return p
    monkeypatch.setattr(s.base, "_prepare_hm_inputs", prepare_inputs)
    monkeypatch.setattr(s.base.hm, "validate_split", lambda data: None)
    monkeypatch.setattr(s.base.hm.hm_graph, "_purchase_keyed", lambda data: None)
    monkeypatch.setattr(s.base.fixed.graph, "centered_edge_signals", lambda *a: {
        "edge_users": p["data"]["pos_key"] // 5, "edge_items": p["data"]["pos_key"] % 5})
    monkeypatch.setattr(s.base.fixed, "build_graph", lambda *a: {
        "audit": {"cv": .3 if mismatch == "graph" else .2}, "beta": .1,
        "weights": np.ones(len(p["data"]["pos_key"]))})
    monkeypatch.setattr(s.base.fixed, "_train", lambda *a, **kw: pytest.fail("training in prepare"))
    if mismatch:
        with pytest.raises(RuntimeError): s.prepare(p["cfg"])
    else:
        result = s.prepare(p["cfg"])
        assert calls == [True]
        assert result["protocol"]["new_fit_models"] == list(s.MODELS)
        assert result["base_cfg"]["SEED_LIST"] == [50]
        assert len(s.report(result)["absolute"]) == 1

