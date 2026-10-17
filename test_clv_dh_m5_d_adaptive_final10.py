from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

import clv_dh_m5_d_adaptive_final10 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared


@pytest.fixture
def batch(prepared):
    refs = {}
    for seed in s.SEEDS:
        ref = prepared["references"]
        refs[seed] = {**ref, "absolute": ref["absolute"].assign(seed=seed),
                      "diagnostics": ref["diagnostics"].assign(seed=seed)}
    prepared["protocol"] = {"code_version": s.VERSION}
    for name in ("protocol.json", "reference_audit.json"):
        s.base.io._atomic_json(prepared["run_dir"] / name, {})
    return {"cfg": s.configure(out_dir=str(prepared["run_dir"])), "shared": prepared,
            "references": refs, "run_dir": prepared["run_dir"], "protocol": prepared["protocol"]}


def test_ten_new_fits_and_cached_rerun(batch, monkeypatch):
    trained, evaluated = [], []
    def train(model, ctx, cfg, mid, seed, store, row_weights=None):
        assert row_weights is ctx["observed_weights"]
        assert cfg.epochs == 300 and mid == s.model_id(seed)
        trained.append(seed)
        return [{"epoch": i} for i in range(1, 301)]
    def evaluate(model, ctx):
        evaluated.append(ctx["cfg"].seeds[0])
        return dict.fromkeys(ctx["references"]["metrics"], 1.02)
    monkeypatch.setattr(s.base.fixed, "_train", train)
    monkeypatch.setattr(s.base.fixed, "_evaluate", evaluate)
    before = s.report(batch)
    assert not before["complete"] and len(before["absolute"]) == 50 and "summary" not in before
    with pytest.raises(RuntimeError, match="未|미완료"):
        s.export_zip(batch, batch["run_dir"] / "early.zip")
    result = s.run(batch)
    assert result["complete"] and len(result["absolute"]) == 60
    assert len(result["summary"]) == 6*142
    assert len(result["comparison"]) == 5*142
    assert len(result["paired_comparison"]) == 5*142*10
    s.run(batch)
    assert trained == list(s.SEEDS) and evaluated == list(s.SEEDS)
    output = s.export_zip(batch, batch["run_dir"] / "done.zip")
    with ZipFile(output) as z:
        assert len([n for n in z.namelist() if n.startswith("arms/")]) == 10
        assert "summary.csv" in z.namelist()


def test_joint_three_component_gradient(batch):
    ctx = s.context(batch, 48)
    model = s.base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=3, n_items=5, features=ctx["features"], adj=ctx["observed_adj"],
        id_dim=8, n_layers=2, pref_reg=.001, shared_l2=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([[4], [4], [0]])
    w = torch.tensor(ctx["observed_weights"][:3], dtype=torch.float32)
    loss, _, _ = s.base.fixed.recheck._batch_loss(model, u, pos, neg, w)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.axis_logits.grad.norm() > 0 and model.rho_logit.grad.abs() > 0
    assert all(layer.weight.grad is not None for layer in model.encoders.values())
    optimizer.step()


def test_no_test_evaluation_before_300(batch, monkeypatch):
    monkeypatch.setattr(s.base.fixed, "_train", lambda *a, **kw: [{"epoch": 299}])
    monkeypatch.setattr(s.base.fixed, "_evaluate", lambda *a: pytest.fail("early test evaluation"))
    with pytest.raises(RuntimeError, match="300epoch"):
        s.run_arm(s.context(batch, 42))


def test_completed_identity_and_frozen_config(batch):
    ctx = s.context(batch, 42)
    path = batch["run_dir"] / "arms" / f"{s.model_id(42)}.json"
    s.base.io._atomic_json(path, {"identity": {}, "epochs": 300})
    with pytest.raises(RuntimeError, match="신원"):
        s.run_arm(ctx)
    with pytest.raises(ValueError):
        s.run_arm({**ctx, "cfg": replace(ctx["cfg"], epochs=100)})
    with pytest.raises(ValueError):
        s.seed_config(batch["cfg"], 99)


def test_aggregation_pairs_seeds_not_row_order():
    metrics = [f"metric_{i}" for i in range(142)]
    rows = [{"model": m, "seed": seed, **dict.fromkeys(metrics, float(seed-40)+(m=="M5_D"))}
            for m in s.ORDER for seed in s.SEEDS]
    a = pd.DataFrame(rows).sample(frac=1, random_state=7)
    summary, comp, paired = s.aggregate(a, metrics)
    row = summary[(summary.model=="M5_D") & (summary.metric==metrics[0])].iloc[0]
    assert row["std"] == pytest.approx(np.arange(3, 13).std(ddof=1))
    assert row["mean"] == 7.5
    assert set(paired.delta) == {1.}
    assert np.allclose(comp.ratio_of_means_pct, 100*(7.5/6.5-1))
    assert all(t.empty for t in s.aggregate(a.iloc[:-1], metrics))
    with pytest.raises(RuntimeError, match="중복"):
        s.aggregate(pd.concat([a, a.iloc[:1]]), metrics)


def test_bad_reference_stops_before_data_prep(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base.prior, "prepare", lambda *a: pytest.fail("unexpected preparation"))
    with pytest.raises(RuntimeError, match="원본"):
        s.prepare(s.configure(m2_reference=str(tmp_path)))


def test_prepare_only_builds_inputs_and_writes_m5d_protocol(batch, monkeypatch):
    shared = batch["shared"]
    shared["data"].update(data_stats={"test": "lastweek"}, splits={"test": ()},
                          train={"b_raw": []}, pos_key=shared["features"]["keys"])
    shared.update(q_n=None, q_v=None, clv_valid=None, base_cfg={},
                  observed_weight_audit={"sha256": "weights"}, observed_graph={"audit": {}})
    p2 = {"input_hash": shared["input_hash"], "data_stats": shared["data"]["data_stats"],
          "feature_sha256": shared["feature_sha256"], "split": "train DAY 1-704; test DAY 705-711"}
    for ref in batch["references"].values():
        ref["sources"] = {"m2": {"protocol": p2}, "m4": {"protocol": {
            "fixed": {}, "m4_c_weight_audit": {"sha256": "weights"}}}}
    monkeypatch.setattr(s, "load_references", lambda cfg: batch["references"])
    calls = []
    def prepare_inputs(cfg):
        calls.append(cfg)
        return shared
    monkeypatch.setattr(s.base.prior, "prepare", prepare_inputs)
    monkeypatch.setattr(s.base.m2, "build_features", lambda *a, **kw: shared["features"])
    monkeypatch.setattr(s.base.fixed, "_train", lambda *a, **kw: pytest.fail("prepare trained a model"))
    result = s.prepare(batch["cfg"])
    assert len(calls) == 1
    assert len(result["protocol"]["new_fit_models"]) == 10
    assert all("m5_d_" in m for m in result["protocol"]["new_fit_models"])
    assert result["protocol"]["config"]["seeds"] == list(s.SEEDS)
    assert result["shared"]["base_cfg"]["SEED_LIST"] == list(s.SEEDS)
    assert len(s.report(result)["absolute"]) == 50
