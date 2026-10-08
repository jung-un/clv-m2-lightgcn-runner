"""Scoped CPU checks, not a substitute for the requested GPU experiment."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import clv_m5_shared_attributes_seed48_screen as s
from clv_m5_shared_attributes_model import JointAttributeLightGCN, build_signals


def toy():
    train = pd.DataFrame({"u_idx": [0, 0, 0, 1, 1, 2], "i_idx": [0, 0, 1, 1, 2, 3],
        "i_raw": [10, 10, 11, 11, 12, 13], "b_raw": [1, 2, 2, 3, 4, 5],
        "t": [650, 680, 683, 675, 680, 680], "v": [2., 2., 4., 4., 6., 1.],
        "up": [2., 2., 4., 4., 6., 1.]})
    products = pd.DataFrame({"PRODUCT_ID": [10, 11, 12, 13],
        "COMMODITY_DESC": ["food", "food", "food", "other"],
        "SUB_COMMODITY_DESC": ["a", "a", "b", "c"]})
    signals = build_signals(train, products, 3, 4)
    cfg = s.configure()
    prep = {"data": {"n_users": 3, "n_items": 4, "train": train,
        "tr_u": train.u_idx.to_numpy(), "tr_i": train.i_idx.to_numpy(),
        "pos_key": np.array([0, 1, 5, 6, 11]), "splits": {"test": ({0: np.array([2])}, {})}},
        "base_cfg": {"TIME_CUTOFF": 690, "EVAL_HOLDOUT": False, "HOLDOUT_DAYS": 0,
            "GRAPH_MODE": "binary", "LOSS_MODE": "plain", "NEG_MODE": "uniform",
            "MIN_ITEM_INTER": 1, "TRAIN_ON_VAL": True, "TEST_DAYS": 7},
        "signals": signals, "q_n": np.array([.2, .8, 0]), "q_v": np.array([.3, .7, 0]),
        "clv_valid": np.array([True, True, False]), "input_hash": "input", "revision": "source",
        "config_hash": "config", "out_dir": Path(cfg.out_dir),
        "row_weights": np.array([.8, .8, 1.4, .9, 1.1, 1.]), "preflight": {"config": asdict(cfg)}}
    return train, products, signals, prep


def model(reg=0):
    _, _, signals, prep = toy()
    torch.manual_seed(48)
    return JointAttributeLightGCN(n_users=3, n_items=4, signals=signals,
        q_n=prep["q_n"], q_v=prep["q_v"], valid=prep["clv_valid"], id_dim=4, pref_reg=reg)


def test_train_only_signals_unique_edges_and_price_midrank():
    train, products, signals, _ = toy()
    np.testing.assert_array_equal(signals["edge_users"] * 4 + signals["edge_items"], [0, 1, 5, 6, 11])
    np.testing.assert_allclose(signals["price_features"][:, 0], [.375, .625, .875, .125])
    np.testing.assert_allclose(signals["price_features"][:2, 1], [.25, .75])
    assert signals["audit"]["all_catalogue_items_preserved"]
    assert signals["relations"][0, 0] > signals["relations"][1, 0]
    with pytest.raises(ValueError):
        build_signals(train, pd.concat([products, products.iloc[:1]]), 3, 4)


def test_missing_metadata_or_price_preserves_items():
    train, products, _, _ = toy()
    train.loc[train.i_idx.eq(3), "up"] = 0
    signals = build_signals(train, products.iloc[:3], 3, 4)
    assert len(signals["item_types"]) == 4
    assert signals["audit"]["metadata_missing_items"] == 1
    np.testing.assert_array_equal(signals["price_features"][3], [0, 0, 0])


def test_positive_whole_item_removed_and_singleton_zero():
    net = model()
    attrs = net.attributes()
    profiles, messages = net.history_vectors(attrs)
    loo = net.leave_one_out(profiles, messages, torch.tensor([0, 2]), torch.tensor([0, 3]))
    torch.testing.assert_close(loo[0], messages[1])
    torch.testing.assert_close(loo[1], torch.zeros(16))
    # Repeated rows of item0 do not leave item0 in the unique-item profile.
    torch.testing.assert_close(profiles[0], (messages[0] + messages[1]) / 2)
    with pytest.raises(ValueError):
        net.leave_one_out(profiles, messages, torch.tensor([0]), torch.tensor([2]))


def test_recommendation_gradient_reaches_every_learned_block_without_l2():
    net = model(reg=0)
    users, pos, neg = torch.tensor([0, 0, 1, 1, 2]), torch.tensor([0, 1, 1, 2, 3]), torch.tensor([2, 3, 0, 3, 0])
    loss, _ = net.weighted_bpr_loss(users, pos, neg, torch.tensor([.8, 1.2, .9, 1.1, 1.]))
    loss.backward()
    grads = net.training_gradient_diagnostics()
    assert all(np.isfinite(v) and v > 0 for v in grads.values()), grads
    assert all(p.requires_grad for p in net.parameters())


def test_nv_changes_both_learned_paths_and_invalid_feature_is_identity():
    net = model()
    attrs = net.attributes()
    before, messages = net.history_vectors(attrs)
    edges = net.edge_users.eq(2)
    torch.testing.assert_close(messages[edges], attrs[net.edge_items[edges]])
    graph = net.graph_weights().detach().clone()
    with torch.no_grad():
        net.context[:2] = 1 - net.context[:2]
    after, _ = net.history_vectors(attrs)
    assert not torch.allclose(before[:2], after[:2], rtol=0, atol=1e-8)
    assert not torch.allclose(graph, net.graph_weights(), rtol=0, atol=1e-8)
    assert net.graph_weights().min() >= np.exp(-.2)
    assert net.graph_weights().max() <= np.exp(.2)


def test_differentiable_adjacency_matches_manual_degree_normalization():
    net = model()
    weights = net.graph_weights()
    dense = torch.zeros(7, 7)
    for u, i, w in zip(net.edge_users, net.edge_items, weights):
        dense[u, i + 3] = w
        dense[i + 3, u] = w
    degree = dense.sum(1)
    expected = dense / (degree[:, None] * degree[None, :]).clamp_min(1e-12).sqrt()
    torch.testing.assert_close(net.weighted_adjacency().to_dense(), expected)
    assert net.weighted_adjacency().values().requires_grad


@pytest.mark.parametrize("field,value", [("seeds", (42,)), ("epochs", 100), ("eta", .2)])
def test_fixed_seed_budget_and_parameters(field, value):
    with pytest.raises(ValueError):
        s.validate_config(replace(s.configure(), **{field: value}))


def test_protected_split_and_train_pair_guards():
    _, _, _, prep = toy()
    s.validate_prepared(prep)
    prep["data"]["splits"]["test"] = ({0: np.array([0])}, {})
    with pytest.raises(RuntimeError, match="학습쌍"):
        s.validate_prepared(prep)
    prep["data"]["splits"] = {"val": ({}, {})}
    with pytest.raises(RuntimeError):
        s.validate_prepared(prep)


def test_missing_m1_stops_before_data_or_training(tmp_path):
    cfg = s.configure(baseline_json=str(tmp_path / "missing.json"))
    with patch.object(s.recheck, "_prepare") as preparation, pytest.raises(FileNotFoundError):
        s.prepare(cfg)
    preparation.assert_not_called()


def test_one_arm_shared_loop_resume_and_cache(tmp_path):
    _, _, _, prep = toy()
    prep["out_dir"] = tmp_path
    cfg = replace(s.configure(out_dir=str(tmp_path)), epochs=2, eval_every=1, id_dim=4, batch_size=3)
    metrics = {m: 1. for m in (*s.baseline.ACCURACY, *s.baseline.ECONOMIC)}
    with patch.object(s.v3, "DEVICE", torch.device("cpu")), \
         patch.object(s.capacity, "_evaluate", return_value=metrics), \
         patch.object(s.capacity, "_clv_score_share", return_value={"clv_score_share": .1}):
        arm = s.run_arm(cfg, prep)
        assert arm["curve"][-1]["epoch"] == 2
        assert arm["final_diagnostics"]["nv_graph_gradient_norm"] > 0
        assert "clv_score_share" not in arm["curve"][-1]["attribute_score_split"]
        identity = s.RunIdentity(s.CODE_VERSION, s.MODEL_ID, 48, "config", "source", "input")
        store = s.ProgressStore(tmp_path / "progress/config", identity)
        restored = s.build_model(cfg, prep)
        optimizer = torch.optim.Adam(restored.parameters(), lr=cfg.lr)
        state = store.restore_epoch(restored, optimizer, np.random.default_rng(48))
        assert state["next_epoch"] == 3
        assert optimizer.state  # one optimizer retains all joint-block state
        assert len(state["history"]) == 2
        # Real fixed config is300; cached result2 must not be accepted as complete.
        with pytest.raises(RuntimeError):
            s.run_arm(cfg, prep)


def test_full_reporting_adverse_results_and_metric_guards(tmp_path):
    cfg = s.configure(out_dir=str(tmp_path))
    metrics = {m: 1. for m in (*s.baseline.ACCURACY, *s.baseline.ECONOMIC)}
    metrics.update({"고CLV_recall@10": 1., "고CLV_price_purchase_amount_weighted_hit@50": 1.})
    def curve(factor):
        return [dict(epoch=e, loss=.1, p_correct=.8,
            metrics={m: v * factor for m, v in metrics.items()}) for e in range(25, 301, 25)]
    _, _, _, prep = toy()
    prep["baseline"] = {"model_id": s.baseline.M1, "curve": curve(1)}
    prep["baseline_provenance"] = {"sha256": "baseline"}
    arm = {"model_id": s.MODEL_ID, "curve": curve(1.02)}
    arm["curve"][-1]["metrics"]["고CLV_price_purchase_amount_weighted_hit@50"] = .8
    result = s.report(cfg, prep, arm)
    assert len(result["absolute"]) == 24
    assert len(result["comparison"]) == 12 * len(metrics)
    assert result["reading"]["exploratory_screen_rule_passed"]
    row = result["comparison"].query("epoch==300 and metric=='고CLV_price_purchase_amount_weighted_hit@50'").iloc[0]
    assert row.relative_percent == pytest.approx(-20.)
    assert not result["reading"]["significance_claim"]
    assert json.loads(Path(result["paths"]["json"]).read_text())["arms"][1]["model_id"] == s.MODEL_ID
    arm["curve"][-1]["metrics"]["recall@50"] = .98
    assert not s.report(cfg, prep, arm)["reading"]["exploratory_screen_rule_passed"]
    arm["curve"][-1]["metrics"].pop("ndcg@10")
    with pytest.raises(RuntimeError):
        s.report(cfg, prep, arm)
