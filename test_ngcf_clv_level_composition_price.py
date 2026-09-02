import numpy as np
import pandas as pd
import pytest
import torch

from ngcf_clv_level_composition_price_model import (
    NGCFCLVLevelCompositionPrice,
)
import ngcf_clv_level_composition_price_screen as runner


def _adj(n_users=3, n_items=4):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(
        indices, values, (n_users + n_items,) * 2
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _clv_model():
    torch.manual_seed(7)
    return NGCFCLVLevelCompositionPrice(
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


def test_id_and_clv_arms_have_declared_layer_and_output_dimensions():
    id64 = NGCFCLVLevelCompositionPrice(
        n_users=3,
        n_items=4,
        adj=_adj(),
        id_dim=64,
        variant="id",
    )
    id67 = NGCFCLVLevelCompositionPrice(
        n_users=3,
        n_items=4,
        adj=_adj(),
        id_dim=67,
        variant="id",
    )
    clv = _clv_model()

    assert id64.layer0_embeddings()[0].shape == (3, 64)
    assert id67.layer0_embeddings()[0].shape == (3, 67)
    assert clv.layer0_embeddings()[0].shape == (3, 9)
    assert id64.embeddings()[0].shape == (3, 64 * 3)
    assert id67.embeddings()[0].shape == (3, 67 * 3)
    assert clv.embeddings()[0].shape == (3, 9 * 3)


def test_clv_coordinates_keep_level_composition_and_fixed_price_direction():
    model = _clv_model()
    user = model.user_auxiliary_coordinates()
    item = model.item_auxiliary_coordinates()

    expected_user_norm = model.q_c * torch.sqrt(
        0.75 + 0.25 * (2.0 * model.q_v - 1.0).pow(2)
    )
    torch.testing.assert_close(
        user.norm(dim=1), expected_user_norm, atol=1e-6, rtol=0
    )
    assert user[0, 1] > 0
    assert user[2, 1] < 0
    assert user[0, 2] < 0
    assert user[2, 2] > 0
    torch.testing.assert_close(
        item[:, :2].norm(dim=1),
        torch.full((4,), np.sqrt(0.75)),
        atol=1e-6,
        rtol=0,
    )


def test_ngcf_uses_two_message_paths_and_concatenates_layer0_1_2():
    model = _clv_model()
    with torch.no_grad():
        for layer in [*model.sum_layers, *model.bi_layers]:
            layer.weight.zero_()
            layer.bias.zero_()
        model.sum_layers[0].weight.copy_(torch.eye(model.input_dim))
        model.bi_layers[0].weight.copy_(torch.eye(model.input_dim))

    user0, item0 = model.layer0_embeddings()
    combined0 = torch.cat([user0, item0])
    neighbour = torch.sparse.mm(model.adj, combined0)
    expected1 = torch.nn.functional.normalize(
        torch.nn.functional.leaky_relu(
            combined0 + neighbour + combined0 * neighbour,
            negative_slope=0.2,
        ),
        dim=1,
    )
    user, item = model.propagated_embeddings()
    combined = torch.cat([user, item])

    torch.testing.assert_close(combined[:, : model.input_dim], combined0)
    torch.testing.assert_close(
        combined[:, model.input_dim : 2 * model.input_dim], expected1
    )
    assert combined.shape[1] == model.input_dim * 3


def test_one_bpr_updates_id_ngcf_and_item_economic_paths():
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
    assert gradients["ngcf_sum_layer0_gradient_norm"] > 0
    assert gradients["ngcf_bi_layer0_gradient_norm"] > 0
    assert gradients["item_relation_projection_gradient_norm"] > 0
    assert gradients["item_price_mixer_gradient_norm"] > 0
    assert gradients["ngcf_auxiliary_input_column_gradient_norm"] > 0


def test_m3_m4_and_external_score_paths_are_rejected():
    model = _clv_model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(users, positives, negatives, weights=torch.tensor([2.0]))
    with pytest.raises(ValueError, match="외부"):
        model.bpr_loss(users, positives, negatives, lam=0.1)


def test_preflight_fixes_development_protocol_and_four_arms():
    cfg = runner.configure_ngcf_clv_screen(
        out_dir="/tmp/ngcf-clv-test",
        baseline_result_dir="/tmp/ngcf-clv-baseline",
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        runner.NGCF_M1_64,
        runner.NGCF_M1_67,
        runner.NGCF_M2,
        runner.NGCF_SHUFFLE,
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


def test_arm_specs_separate_capacity_and_attribution_controls():
    assert runner._arm_spec(runner.NGCF_M1_64) == ("id", 64, "none")
    assert runner._arm_spec(runner.NGCF_M1_67) == ("id", 67, "none")
    assert runner._arm_spec(runner.NGCF_M2) == ("clv", 64, "observed")
    assert runner._arm_spec(runner.NGCF_SHUFFLE) == (
        "clv",
        64,
        "degree_matched_shuffle",
    )


def test_degree_matched_shuffle_keeps_each_joint_tuple_inside_degree_stratum():
    n_users = 12
    rows = []
    for user in range(n_users):
        for item in range(user + 1):
            rows.append({"u_idx": user, "i_idx": item})
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
    cfg = runner.NGCFCLVScreenConfig(
        shuffle_degree_bins=3,
        shuffle_seed=7,
    )

    shuffled = runner.source._degree_matched_clv_shuffle(prepared, cfg)
    source_user = shuffled["source_user"]

    assert shuffled["changed_valid_user_share"] > 0
    np.testing.assert_array_equal(shuffled["q_n"], q_n[source_user])
    np.testing.assert_array_equal(shuffled["q_v"], q_v[source_user])
    np.testing.assert_array_equal(shuffled["q_c"], q_c[source_user])
    np.testing.assert_array_equal(
        shuffled["stratum"], shuffled["stratum"][source_user]
    )


def test_screening_requires_capacity_and_shuffle_attribution_together():
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
        runner.NGCF_M1_67: common,
        runner.NGCF_M2: {key: value * 1.01 for key, value in common.items()},
        runner.NGCF_SHUFFLE: {key: value * 1.005 for key, value in common.items()},
    }
    assert runner.screening_reading(rows)["positive_screen"] is True

    rows[runner.NGCF_SHUFFLE] = {
        key: value * 1.02 for key, value in common.items()
    }
    reading = runner.screening_reading(rows)
    assert reading["positive_screen"] is False
    assert reading["high_clv_better_than_ngcf_m1_67"] is True
    assert reading["high_clv_better_than_degree_matched_shuffle"] is False
