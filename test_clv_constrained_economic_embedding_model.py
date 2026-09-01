import numpy as np
import pandas as pd
import pytest
import torch

from clv_constrained_economic_embedding_model import (
    ConstrainedCLVEconomicLightGCN,
)
import lightgcn_clv_constrained_economic_embedding as runner


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


def _model(*, rho=0.05, layers=1):
    torch.manual_seed(11)
    return ConstrainedCLVEconomicLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.9, 0.5, 0.1], np.float32),
        q_v=np.array([0.1, 0.5, 0.9], np.float32),
        q_c=np.array([0.8, 0.5, 0.2], np.float32),
        user_clv_valid=np.ones(3, bool),
        item_economic_features=np.array(
            [[-1.0, -0.8], [-0.3, 0.1], [0.4, 0.5], [1.0, 0.9]],
            np.float32,
        ),
        item_economic_valid=np.ones(4, bool),
        adj=_adj(),
        id_dim=6,
        clv_dim=3,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_user_coordinates_explicitly_encode_level_composition_and_value():
    model = _model(layers=0)
    user = model.clv_user_embeddings()

    expected_norm = model.q_c * torch.sqrt(
        0.75 + 0.25 * (2.0 * model.q_v - 1.0).pow(2)
    )
    torch.testing.assert_close(
        user.norm(dim=1), expected_norm, atol=1e-6, rtol=0
    )
    assert user[0, 1] > 0  # N-dominant composition
    assert user[2, 1] < 0  # V-dominant composition
    assert user[0, 2] < 0  # low-V price preference
    assert user[2, 2] > 0  # high-V price preference


def test_item_clv_block_uses_two_id_coordinates_and_one_positive_price_mix():
    model = _model(layers=0)
    assert not hasattr(model, "item_response")
    with torch.no_grad():
        model.item_collaborative_projection.weight.zero_()
        model.item_collaborative_projection.weight[0, 0] = 1.0
        model.item_collaborative_projection.weight[1, 1] = 1.0

    item = model.clv_item_embeddings()

    expected_price = (
        model.item_economic_features.mean(dim=1, keepdim=True) * np.sqrt(0.25)
    )
    torch.testing.assert_close(item[:, 2:], expected_price)
    torch.testing.assert_close(
        item[:, :2].norm(dim=1),
        torch.full((4,), np.sqrt(0.75)),
        atol=1e-6,
        rtol=0,
    )
    assert torch.all(item.norm(dim=1) <= 1.0 + 1e-6)


def test_layer0_is_id_plus_one_bounded_three_dimensional_block():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 9)
    assert item.shape == (4, 9)
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    assert torch.all(user[:, 6:].norm(dim=1) <= np.sqrt(0.05) + 1e-6)
    assert torch.all(item[:, 6:].norm(dim=1) <= np.sqrt(0.05) + 1e-6)


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def test_one_bpr_trains_id_relation_projection_and_positive_price_mix():
    model = _model(layers=1)
    users = torch.tensor([0, 1, 2])
    positives = torch.tensor([0, 1, 3])
    negatives = torch.tensor([2, 3, 0])

    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.item_collaborative_projection.weight.grad.abs().sum() > 0
    assert model.item_price_logits.grad.abs().sum() > 0


def test_relation_and_price_views_partition_the_auxiliary_coordinates():
    model = _model(layers=1)
    full_user, full_item = model.propagated_embeddings()
    relation_user, relation_item = model.component_embeddings("relation")
    price_user, price_item = model.component_embeddings("price")
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(relation_user[:, 8:], torch.zeros_like(relation_user[:, 8:]))
    torch.testing.assert_close(relation_item[:, 8:], torch.zeros_like(relation_item[:, 8:]))
    torch.testing.assert_close(price_user[:, 6:8], torch.zeros_like(price_user[:, 6:8]))
    torch.testing.assert_close(price_item[:, 6:8], torch.zeros_like(price_item[:, 6:8]))
    torch.testing.assert_close(full_user[:, :6], id_user)
    torch.testing.assert_close(full_item[:, :6], id_item)


def test_m3_m4_and_external_score_paths_are_rejected():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            users, positives, negatives, None, 0.0, torch.tensor([2.0])
        )
    with pytest.raises(ValueError, match="외부"):
        model.bpr_loss(users, positives, negatives, None, 0.1, None)


def test_rho10_attribution_config_is_fixed_before_execution():
    cfg = runner.configure_rho10_attribution_run(
        out_dir="/tmp/rho10",
        baseline_result_dir="/tmp/baseline",
    )

    assert cfg.rho == 0.10
    assert cfg.item_price_budget == 0.25
    assert cfg.include_degree_matched_shuffle is True
    assert cfg.shuffle_degree_bins == 10
    assert runner.preflight_summary(cfg)["code_version"] == runner.RHO10_CODE_VERSION


def test_degree_matched_shuffle_moves_joint_clv_tuples_only_within_strata():
    n_users = 12
    rows = []
    for user in range(n_users):
        for item in range(user + 1):
            rows.append({"u_idx": user, "i_idx": item})
    q_n = np.linspace(0.05, 0.95, n_users, dtype=np.float32)
    q_v = np.linspace(0.95, 0.05, n_users, dtype=np.float32)
    q_c = np.arange(n_users, dtype=np.float32) / n_users
    prepared = {
        "data": {
            "train": pd.DataFrame(rows),
            "n_users": n_users,
        },
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": np.ones(n_users, dtype=bool),
    }
    cfg = runner.ConstrainedEconomicConfig(
        shuffle_degree_bins=3,
        shuffle_seed=7,
    )

    shuffled = runner._degree_matched_clv_shuffle(prepared, cfg)
    source = shuffled["source_user"]

    assert shuffled["changed_valid_user_share"] > 0.0
    np.testing.assert_array_equal(shuffled["q_n"], q_n[source])
    np.testing.assert_array_equal(shuffled["q_v"], q_v[source])
    np.testing.assert_array_equal(shuffled["q_c"], q_c[source])
    np.testing.assert_array_equal(
        shuffled["stratum"], shuffled["stratum"][source]
    )
