import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from graphsage_clv_level_composition_price_model import (
    GraphSAGECLVLevelCompositionPrice,
    GraphSAGEMeanLayer,
)
import graphsage_clv_level_composition_price_screen as runner


def _adj(n_users=3, n_items=4, value=1.0):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.full((len(rows),), value, dtype=torch.float32)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(
            indices,
            values,
            (n_users + n_items,) * 2,
        ).coalesce()


def _clv_model():
    torch.manual_seed(7)
    return GraphSAGECLVLevelCompositionPrice(
        n_users=3,
        n_items=4,
        adj=_adj(),
        id_dim=6,
        variant="clv",
        q_n=np.array([0.9, 0.5, 0.1], np.float32),
        q_v=np.array([0.1, 0.5, 0.9], np.float32),
        q_c=np.array([0.8, 0.5, 0.2], np.float32),
        user_clv_valid=np.ones(3, bool),
        item_economic_features=np.array(
            [[-1.0, -0.8], [-0.3, 0.1], [0.4, 0.5], [1.0, 0.9]],
            np.float32,
        ),
        item_economic_valid=np.ones(4, bool),
        rho=0.05,
        item_price_budget=0.25,
        n_layers=2,
        pref_reg=1e-4,
    )


def test_mean_layer_uses_binary_neighbor_mean_and_explicit_self_path():
    layer = GraphSAGEMeanLayer(2)
    with torch.no_grad():
        layer.projection.weight.zero_()
        layer.projection.bias.zero_()
        layer.projection.weight[:, :2].copy_(torch.eye(2))
        layer.projection.weight[:, 2:].copy_(torch.eye(2))

    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 1]])
    with torch.sparse.check_sparse_tensor_invariants(False):
        mean_adjacency = torch.sparse_coo_tensor(
            indices,
            torch.tensor([0.5, 0.5, 1.0, 1.0]),
            (3, 3),
        ).coalesce()
    output = layer(features, mean_adjacency)
    neighbor_mean = torch.tensor([[4.0, 5.0], [1.0, 2.0], [3.0, 4.0]])
    expected = F.normalize(features + neighbor_mean, p=2, dim=1)

    torch.testing.assert_close(output, expected)


def test_graph_values_do_not_change_binary_mean_aggregation():
    torch.manual_seed(3)
    first = GraphSAGECLVLevelCompositionPrice(
        n_users=3, n_items=4, adj=_adj(value=1.0), id_dim=5, variant="id"
    )
    second = GraphSAGECLVLevelCompositionPrice(
        n_users=3, n_items=4, adj=_adj(value=7.0), id_dim=5, variant="id"
    )
    second.load_state_dict(first.state_dict())

    first_user, first_item = first.propagated_embeddings()
    second_user, second_item = second.propagated_embeddings()

    torch.testing.assert_close(first_user, second_user)
    torch.testing.assert_close(first_item, second_item)


def test_id_and_clv_arms_keep_one_mean_aggregated_embedding_space():
    id64 = GraphSAGECLVLevelCompositionPrice(
        n_users=3, n_items=4, adj=_adj(), id_dim=64, variant="id"
    )
    id67 = GraphSAGECLVLevelCompositionPrice(
        n_users=3, n_items=4, adj=_adj(), id_dim=67, variant="id"
    )
    clv = _clv_model()

    assert id64.layer0_embeddings()[0].shape == (3, 64)
    assert id67.layer0_embeddings()[0].shape == (3, 67)
    assert clv.layer0_embeddings()[0].shape == (3, 9)
    assert id64.embeddings()[0].shape == (3, 64)
    assert id67.embeddings()[0].shape == (3, 67)
    assert clv.embeddings()[0].shape == (3, 9)


def test_two_graphsage_layers_are_mean_aggregated_with_layer0():
    model = _clv_model()
    user0, item0 = model.layer0_embeddings()
    combined0 = torch.cat([user0, item0])
    layers = model.propagation_layers()
    user, item = model.propagated_embeddings()
    combined = torch.cat([user, item])

    assert len(layers) == 3
    torch.testing.assert_close(layers[0], combined0)
    torch.testing.assert_close(combined, torch.stack(layers).mean(dim=0))


def test_one_bpr_updates_id_graphsage_and_item_economic_paths():
    model = _clv_model()
    users = torch.tensor([0, 1, 2])
    positives = torch.tensor([0, 1, 3])
    negatives = torch.tensor([2, 3, 0])

    loss, diagnostics = model.bpr_loss(users, positives, negatives)
    loss.backward()
    gradients = model.training_gradient_diagnostics()

    assert diagnostics["objective"] == "plain_bpr"
    assert gradients["id_user_gradient_norm"] > 0
    assert gradients["id_item_gradient_norm"] > 0
    assert gradients["graphsage_projection_layer0_gradient_norm"] > 0
    assert gradients["graphsage_auxiliary_input_column_gradient_norm"] > 0
    assert gradients["item_relation_projection_gradient_norm"] > 0
    assert gradients["item_price_mixer_gradient_norm"] > 0


def test_m3_m4_and_external_score_paths_are_rejected():
    model = _clv_model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(users, positives, negatives, weights=torch.tensor([2.0]))
    with pytest.raises(ValueError, match="외부"):
        model.bpr_loss(users, positives, negatives, lam=0.1)


def test_preflight_fixes_m2_boundaries_and_four_graphsage_arms():
    cfg = runner.configure_graphsage_clv_screen(
        out_dir="/tmp/graphsage-clv-test",
        baseline_result_dir="/tmp/graphsage-clv-baseline",
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        runner.GRAPHSAGE_M1_64,
        runner.GRAPHSAGE_M1_67,
        runner.GRAPHSAGE_M2,
        runner.GRAPHSAGE_SHUFFLE,
    ]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "new_item_task": True,
        "train_pairs_excluded_from_truth_and_ranking": True,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["min_item_interactions"] == 1


def test_arm_specs_separate_capacity_and_clv_assignment_controls():
    assert runner._arm_spec(runner.GRAPHSAGE_M1_64) == ("id", 64, "none")
    assert runner._arm_spec(runner.GRAPHSAGE_M1_67) == ("id", 67, "none")
    assert runner._arm_spec(runner.GRAPHSAGE_M2) == ("clv", 64, "observed")
    assert runner._arm_spec(runner.GRAPHSAGE_SHUFFLE) == (
        "clv",
        64,
        "degree_matched_shuffle",
    )


def test_degree_matched_shuffle_keeps_joint_clv_tuple_inside_degree_stratum():
    n_users = 12
    rows = [
        {"u_idx": user, "i_idx": item}
        for user in range(n_users)
        for item in range(user + 1)
    ]
    q_n = np.linspace(0.05, 0.95, n_users, dtype=np.float32)
    q_v = np.linspace(0.95, 0.05, n_users, dtype=np.float32)
    q_c = np.arange(n_users, dtype=np.float32) / n_users
    prepared = {
        "data": {"train": pd.DataFrame(rows), "n_users": n_users},
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": np.ones(n_users, dtype=bool),
    }
    cfg = runner.GraphSAGECLVScreenConfig(shuffle_degree_bins=3, shuffle_seed=7)

    shuffled = runner.source._degree_matched_clv_shuffle(prepared, cfg)
    source_user = shuffled["source_user"]

    assert shuffled["changed_valid_user_share"] > 0
    np.testing.assert_array_equal(shuffled["q_n"], q_n[source_user])
    np.testing.assert_array_equal(shuffled["q_v"], q_v[source_user])
    np.testing.assert_array_equal(shuffled["q_c"], q_c[source_user])
    np.testing.assert_array_equal(
        shuffled["stratum"], shuffled["stratum"][source_user]
    )


def test_screen_requires_accuracy_economic_and_all_attribution_checks():
    common = {
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
        "고CLV_recall@10": 1.0,
        "고CLV_ndcg@10": 1.0,
        "price_purchase_amount_weighted_hit@10": 1.0,
    }
    rows = {
        runner.GRAPHSAGE_M1_67: common,
        runner.GRAPHSAGE_M2: {key: value * 1.01 for key, value in common.items()},
        runner.GRAPHSAGE_SHUFFLE: {
            key: value * 1.005 for key, value in common.items()
        },
    }
    assert runner.screening_reading(rows)["positive_screen"] is True

    rows[runner.GRAPHSAGE_SHUFFLE] = dict(rows[runner.GRAPHSAGE_SHUFFLE])
    rows[runner.GRAPHSAGE_SHUFFLE]["고CLV_ndcg@10"] = 1.02
    reading = runner.screening_reading(rows)
    assert reading["positive_screen"] is False
    assert reading["accuracy_floor_vs_graphsage_m1_67"] is True
    assert reading["high_clv_attribution_pass"] is False
