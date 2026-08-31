import numpy as np
import pytest
import torch

from clv_joint_response_embedding_model import JointCLVResponseLightGCN


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
    torch.manual_seed(7)
    return JointCLVResponseLightGCN(
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
        clv_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_layer0_is_id_plus_one_four_dimensional_joint_clv_block():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 10)
    assert item.shape == (4, 10)
    assert model.total_dim == 10
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    assert torch.all(user[:, 6:].abs() <= np.sqrt(0.05) + 1e-7)
    assert torch.all(item[:, 6:].abs() <= np.sqrt(0.05) + 1e-7)


def test_context_keeps_overall_clv_and_n_v_composition_separate_and_centered():
    model = _model(layers=0)
    context = model.clv_context

    torch.testing.assert_close(context.mean(0), torch.zeros(2), atol=1e-7, rtol=0)
    assert context[0, 0] > context[2, 0]  # higher total CLV
    assert context[0, 1] > 0  # N-dominant
    assert context[2, 1] < 0  # V-dominant


def test_item_block_combines_item_response_and_two_price_positions():
    model = _model(layers=0)
    with torch.no_grad():
        model.item_response.weight.zero_()
        model.item_economic_projection.weight.zero_()
        model.item_economic_projection.weight[0, 0] = 1.0
        model.item_economic_projection.weight[1, 1] = 1.0

    item = model.clv_item_embeddings()

    assert item[0, 0] < 0 < item[3, 0]
    assert item[0, 1] < 0 < item[3, 1]
    torch.testing.assert_close(item.norm(dim=1), torch.ones(4), atol=1e-6, rtol=0)


def test_rho_zero_is_exact_ordinary_lightgcn_with_zero_auxiliary_coordinates():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))
    assert model.representation_diagnostics()["rho_zero_auxiliary_max_abs"] == 0.0


def test_one_plain_bpr_trains_id_and_joint_clv_parameters_together():
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
    assert model.user_clv_projection.weight.grad.abs().sum() > 0
    assert model.item_response.weight.grad.abs().sum() > 0
    assert model.item_economic_projection.weight.grad.abs().sum() > 0


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
