"""Minimum checks for the fixed-graph, matched-seed representation screen."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import clv_m5_shared_attributes_centered_seed48_screen as s
from test_clv_m5_shared_attributes_seed48 import toy, RejectDenseNodeSquare


def model():
    _, _, signals, prepared = toy()
    weights = np.array([.9, 1.1, 1.05, .95, 1.], np.float32)
    adjacency = s.v3.build_adj(signals["edge_users"], signals["edge_items"], weights, 3, 4)
    torch.manual_seed(48)
    return s.FixedGraphAttributeLightGCN(adjacency=adjacency, graph_weights=weights,
        n_users=3, n_items=4, signals=signals, q_n=prepared["q_n"], q_v=prepared["q_v"],
        valid=prepared["clv_valid"], id_dim=4, pref_reg=0)


def test_fixed_graph_propagation_and_no_learned_graph_parameters():
    net = model()
    assert not hasattr(net, "graph_net")
    assert not net.fixed_adj.requires_grad
    assert all(p.requires_grad for p in net.parameters())
    current = torch.cat([net.E_u.weight, net.E_i.weight])
    layers = [current]
    for _ in range(net.n_layers):
        current = net.fixed_adj.to_dense() @ current
        layers.append(current)
    expected = torch.stack(layers).mean(0)
    torch.testing.assert_close(torch.cat(net.id_vectors()), expected)
    assert net.representation_diagnostics()["graph_parameters"] == 0


def test_recommendation_gradient_reaches_m2_without_dense_node_square():
    net = model()
    users, pos, neg = torch.tensor([0, 0, 1, 1, 2]), torch.tensor([0, 1, 1, 2, 3]), torch.tensor([2, 3, 0, 3, 0])
    with RejectDenseNodeSquare(7):
        loss, _ = net.weighted_bpr_loss(users, pos, neg, torch.ones(5))
        loss.backward()
    grads = net.training_gradient_diagnostics()
    assert len(grads) == 5 and all(np.isfinite(v) and v > 0 for v in grads.values())
    optimizer = torch.optim.Adam(net.parameters())
    assert set(map(id, optimizer.param_groups[0]["params"])) == set(map(id, net.parameters()))


def test_same_m2_initialization_history_loo_and_nv_only_feature_path():
    net = model()
    from test_clv_m5_shared_attributes_seed48 import model as old_model
    old = old_model()
    for name in ("E_u.weight", "E_i.weight", "type_encoder.weight", "price_encoder.weight",
                 "price_encoder.bias", "feature_net.0.weight", "feature_net.2.weight"):
        torch.testing.assert_close(net.state_dict()[name], old.state_dict()[name])
    attrs = net.attributes()
    before, messages = net.history_vectors(attrs)
    loo = net.leave_one_out(before, messages, torch.tensor([0, 2]), torch.tensor([0, 3]))
    torch.testing.assert_close(loo[0], messages[1])
    torch.testing.assert_close(loo[1], torch.zeros(16))
    graph = net.fixed_adj.to_dense().clone()
    with torch.no_grad():
        net.context[:2] = 1 - net.context[:2]
    after, _ = net.history_vectors(attrs)
    assert not torch.allclose(before[:2], after[:2], rtol=0, atol=1e-8)
    torch.testing.assert_close(net.fixed_adj.to_dense(), graph)
    torch.testing.assert_close(before[2], after[2])


def test_config_and_split_protection():
    cfg = s.configure()
    for key, value in (("seeds", (42,)), ("epochs", 100), ("eta", .2), ("target_cv", .25)):
        with pytest.raises(ValueError):
            s.validate_config(replace(cfg, **{key: value}))
    with pytest.raises(ValueError):
        s.configure(seed=44)
    _, _, _, prepared = toy()
    s.shared.validate_prepared(prepared)
    bad = deepcopy(prepared)
    bad["data"]["train"]["t"] = 704
    with pytest.raises(RuntimeError):
        s.shared.validate_prepared(bad)
    bad = deepcopy(prepared)
    bad["data"]["splits"]["test"] = ({0: np.array([0])}, {})
    with pytest.raises(RuntimeError):
        s.shared.validate_prepared(bad)


def test_full_report_primary_gate_uses_c_not_m1_and_reports_losses(tmp_path):
    cfg = s.configure(out_dir=str(tmp_path))
    metrics = {key: 1. for key in (*s.ACCURACY, *s.ECONOMIC, "고CLV_recall@10", "coverage@10")}
    def arm(model_id, value):
        return dict(model_id=model_id, curve=[dict(epoch=epoch, metrics={k: value for k in metrics},
            loss=1., p_correct=.5) for epoch in range(25, 301, 25)])
    baseline = arm(s.shared.baseline.M1, .5)
    reference = arm(s.REFERENCE, 1.)
    candidate = arm(s.MODEL_ID, 1.01)
    candidate["curve"][-1]["metrics"]["recall@20"] = .985
    candidate["curve"][-1]["metrics"]["고CLV_recall@10"] = .9
    prepared = dict(baseline=baseline, out_dir=Path(tmp_path), preflight={"config": asdict(cfg)},
        revision="source", input_hash="input", config_hash="config", baseline_provenance={},
        existing_c_diagnostic={}, new_pair_diagnostic={})
    result = s.report(cfg, prepared, [reference, candidate])
    assert not result["reading"]["exploratory_screen_rule_passed"]
    assert not result["reading"]["accuracy_and_economic_guards"]["recall@20"]
    assert result["reading"]["primary_reference"] == s.REFERENCE
    assert len(result["absolute"]) == 36
    assert len(result["comparison"]) == 12 * 3 * len(metrics)
    losses = result["comparison"].query("epoch == 300 and model_id == @s.MODEL_ID and reference == @s.REFERENCE")
    assert losses.set_index("metric").at["고CLV_recall@10", "delta"] < 0


def test_run_diagnostic_then_two_fits_in_order_and_no_m1_fit():
    cfg = s.configure()
    _, _, _, prepared = toy()
    prepared["preflight"]["config"] = asdict(cfg)
    prepared["baseline_provenance"] = {"sha256": "same"}
    events = []
    def fit(cfg, prepared, model_id):
        events.append(model_id)
        return {}, {}, []
    with patch.object(s.shared.baseline, "load_baseline", return_value=({}, prepared["baseline_provenance"])), \
         patch.object(s, "diagnose_existing_c", side_effect=lambda *a: events.append("diagnostic")), \
         patch.object(s, "run_arm", side_effect=fit), \
         patch.object(s, "_save_pair", return_value={}), patch.object(s, "report", return_value={}):
        s.run(cfg, prepared)
    assert events == ["diagnostic", s.REFERENCE, s.MODEL_ID]


def test_c_readback_all_keys_and_only_alignment_has_wider_tolerance():
    recorded = {"recall@10": .02, "user_value_tendency_recommended_price_alignment": .078044}
    measured = {"recall@10": .02, "user_value_tendency_recommended_price_alignment": .078048}
    assert all(row["passed"] for row in s._readback(recorded, measured, "C"))
    measured["recall@10"] += 4e-6
    assert not s._readback(recorded, measured, "C")[0]["passed"]
    with pytest.raises(RuntimeError):
        s._readback(recorded, {"recall@10": .02}, "C")


def test_precheck_requires_existing_development_c_artifacts(tmp_path):
    cfg = s.configure(c_reference_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        s._c_source(cfg)
    report_path = tmp_path / f"{s.m4.CODE_VERSION}_config.json"
    absolute = tmp_path / "absolute.csv"
    absolute.write_text("model_id,seed,epoch\n")
    checkpoint = tmp_path / "progress" / "config" / "resume" / f"m5_m3_m4_user_centered_dev_{s.REFERENCE}_s44_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    payload = dict(seed=44, split=s.shared.SPLIT, final_test=False, holdout=False,
                   config=asdict(cfg), result_paths={"absolute_csv": str(absolute)})
    report_path.write_text(json.dumps(payload))
    assert s._c_source(cfg)["config_hash"] == "config"
    payload["final_test"] = True
    report_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        s._c_source(cfg)


def test_fixed_model_checkpoint_restore_and_completed_cache(tmp_path):
    cfg = s.configure(out_dir=str(tmp_path))
    _, _, _, prepared = toy()
    prepared.update(out_dir=tmp_path, graph={"weight_sha256": "graph"})
    prepared["preflight"]["m4_audit"] = {"sha256": "row"}
    metrics = {"recall@10": .02}
    calls = []
    def train(net, prep, cfg, spec, seed, store, row_weights):
        calls.append(spec["model_id"])
        optimizer = torch.optim.Adam(net.parameters())
        curve = [dict(epoch=300, metrics=metrics, loss=1., p_correct=.5)]
        store.save_epoch(net, optimizer, np.random.default_rng(48), epoch=300, history=curve,
                         wall_clock_sec=1., selection="none")
        return curve
    with patch.object(s, "build_model", side_effect=lambda *a: model()), \
         patch.object(s, "_runtime_root", return_value=tmp_path / "local"), \
         patch.object(s.capacity, "_train_curve", side_effect=train), \
         patch.object(s.capacity, "_evaluate", return_value=metrics), \
         patch.object(s.ranks, "_score_truth", return_value={}):
        first, _, _ = s.run_arm(cfg, prepared, s.MODEL_ID)
        second, _, _ = s.run_arm(cfg, prepared, s.MODEL_ID)
    assert calls == [s.MODEL_ID]  # already completed => no second optimizer loop
    assert first == second
    assert list((tmp_path / "reports").glob("*_readback.csv"))
