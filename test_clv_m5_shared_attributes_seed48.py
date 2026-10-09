"""Scoped CPU checks, not a substitute for the requested GPU experiment."""
from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import clv_m5_shared_attributes_seed48_screen as s
from clv_m5_shared_attributes_model import JointAttributeLightGCN, build_signals, edge_sparse_mm


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


class RejectDenseNodeSquare(TorchDispatchMode):
    """Catch the actual OOM pattern before a huge allocation, on CPU or GPU."""
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (str(func) == "aten.mm.default" and len(args) == 2
                and args[0].shape[0] == self.nodes and args[1].shape[1] == self.nodes):
            raise AssertionError("OOM regression: backward allocates dense node_count x node_count")
        result = func(*args, **(kwargs or {}))
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        for value in outputs:
            if (isinstance(value, torch.Tensor) and value.layout == torch.strided
                    and tuple(value.shape) == (self.nodes, self.nodes)):
                raise AssertionError("OOM regression: dense node_count x node_count result")
        return result


def test_joint_backward_never_materializes_dense_node_square():
    net = model(reg=0)
    loss, _ = net.weighted_bpr_loss(torch.tensor([0, 1]), torch.tensor([0, 1]),
                                  torch.tensor([2, 3]), torch.ones(2))
    with RejectDenseNodeSquare(7):
        loss.backward()
    assert net.training_gradient_diagnostics()["nv_graph_gradient_norm"] > 0


def test_edge_matmul_forward_and_gradcheck_with_duplicate_coordinates():
    indices = torch.tensor([[0, 0, 1, 2, 2], [1, 1, 2, 0, 3]])
    values = torch.tensor([.2, .3, .4, .5, .6], dtype=torch.double, requires_grad=True)
    dense = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    expected = torch.sparse_coo_tensor(indices, values, (5, 5)).to_dense() @ dense
    actual = edge_sparse_mm(indices, values, dense, chunk_size=2)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda v, x: edge_sparse_mm(indices, v, x, 2),
                                    (values, dense))


def test_joint_loss_and_all_parameter_gradients_match_dense_reference():
    net = model(reg=0).double()
    reference = deepcopy(net)
    def dense_id_vectors():
        adj = reference.weighted_adjacency().to_dense()
        current = torch.cat([reference.E_u.weight, reference.E_i.weight])
        layers = [current]
        for _ in range(reference.n_layers):
            current = adj @ current
            layers.append(current)
        return torch.stack(layers).mean(0).split([3, 4])
    args = (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2]), torch.tensor([2, 3, 0]),
            torch.tensor([.8, 1.2, 1.], dtype=torch.double))
    actual, _ = net.weighted_bpr_loss(*args)
    with patch.object(reference, "id_vectors", side_effect=dense_id_vectors):
        expected, _ = reference.weighted_bpr_loss(*args)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.backward()
    expected.backward()
    for (name, param), (other_name, other) in zip(net.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(param.grad, other.grad, rtol=1e-10, atol=1e-12)


def test_edge_backward_large_node_count_uses_only_existing_edges():
    n = 100000  # dense float32 adjacency would require37.3GiB
    indices = torch.tensor([[0, 1, n - 1], [1, n - 1, 0]])
    values = torch.ones(3, requires_grad=True)
    dense = torch.randn(n, 4, requires_grad=True)
    with RejectDenseNodeSquare(n):
        edge_sparse_mm(indices, values, dense, chunk_size=2).square().sum().backward()
    assert values.grad.shape == (3,)
    assert dense.grad.shape == (n, 4)
    assert torch.isfinite(values.grad).all()


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


def test_memory_fix_reuses_inputs_without_preparation_and_preserves_old_run(tmp_path):
    _, _, _, prep = toy()
    prep["out_dir"] = tmp_path / "old"
    prep["preflight"]["metadata_sha256"] = "metadata"
    prep["baseline_provenance"] = {"sha256": "baseline"}
    cfg = s.configure(out_dir=str(tmp_path / "fixed"))
    with patch.object(s.baseline, "load_baseline", return_value=({}, prep["baseline_provenance"])), \
         patch.object(s.capacity.moe, "source_revision", return_value="fixed-source"), \
         patch.object(s.recheck, "_prepare") as preparation:
        ready = s.reuse_prepared_after_memory_fix(cfg, prep)
    preparation.assert_not_called()
    assert ready["data"] is prep["data"]
    assert ready["signals"] is prep["signals"]
    assert ready["row_weights"] is prep["row_weights"]
    assert prep["revision"] == "source"
    assert ready["revision"] == "fixed-source"
    assert ready["config_hash"] != prep["config_hash"]
    assert ready["out_dir"] != prep["out_dir"]
    old = prep["out_dir"] / "progress/config/resume/m5_s48_latest.pt"
    old.parent.mkdir(parents=True)
    old.touch()
    with pytest.raises(RuntimeError, match="checkpoint"):
        s.reuse_prepared_after_memory_fix(cfg, prep)


def test_local_first_progress_writes_drive_only_at_durable_boundary(tmp_path):
    identity = s.RunIdentity(s.CODE_VERSION, s.MODEL_ID, 48, "config", "source", "input")
    local = tmp_path / "local"
    durable = tmp_path / "drive"
    store = s.LocalFirstProgressStore(
        local, durable, identity, max_epoch=3, durable_every=2,
    )
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(48)

    store.mark_stage("running", epoch=0, max_epoch=3)
    store.heartbeat(epoch=1, max_epoch=3, batch=1, batches=2)
    assert not durable.exists()
    store.save_epoch(net, optimizer, rng, epoch=1, history=[])
    assert store.latest_checkpoint.is_file()
    assert not store.durable_checkpoint.exists()
    store.save_epoch(net, optimizer, rng, epoch=2, history=[])
    assert store.durable_checkpoint.is_file()
    assert store.durable_state.is_file()
    assert not (durable / "progress.json").exists()
    assert not (durable / "progress.csv").exists()

    recovered = s.LocalFirstProgressStore(
        tmp_path / "new-local", durable, identity, max_epoch=3, durable_every=2,
    )
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    state = recovered.restore_epoch(restored, restored_optimizer, np.random.default_rng(48))
    assert state["next_epoch"] == 3


def test_drive_io_fix_migrates_exact_previous_checkpoint_without_deleting_it(tmp_path):
    old_identity = s.RunIdentity(
        s.PREVIOUS_CODE_VERSION, s.MODEL_ID, 48, "old-config", "old-source", "input",
    )
    old_store = s.ProgressStore(tmp_path / "old-drive", old_identity)
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    old_store.save_epoch(net, optimizer, np.random.default_rng(48), epoch=7, history=[])
    old_sha = s.file_sha256(old_store.latest_checkpoint)

    new_identity = s.RunIdentity(
        s.CODE_VERSION, s.MODEL_ID, 48, "new-config", "new-source", "input",
    )
    migrated = s.LocalFirstProgressStore(
        tmp_path / "new-local", tmp_path / "new-drive", new_identity,
        max_epoch=300, resume_source=old_store.latest_checkpoint,
        resume_source_identity=asdict(old_identity),
    )
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    state = migrated.restore_epoch(
        restored, restored_optimizer, np.random.default_rng(48),
    )
    assert state["next_epoch"] == 8
    assert s.file_sha256(old_store.latest_checkpoint) == old_sha
    payload = torch.load(migrated.latest_checkpoint, map_location="cpu", weights_only=False)
    assert payload["identity"] == asdict(new_identity)


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
