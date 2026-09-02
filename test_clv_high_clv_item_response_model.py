import numpy as np
import pytest
import torch

from clv_high_clv_item_response_model import HighCLVItemResponseLightGCN


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


def _model(*, rho=0.05, layers=2):
    torch.manual_seed(19)
    return HighCLVItemResponseLightGCN(
        n_users=3,
        n_items=4,
        high_clv_gate=np.array([1.0, 0.0, 0.0], np.float32),
        adj=_adj(),
        id_dim=6,
        response_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_auxiliary_user_has_no_free_embedding_and_is_history_derived():
    model = _model()

    assert not hasattr(model, "response_user")
    assert model.item_response.weight.shape == (4, 4)
    response_user, response_item = model.response_embeddings()

    assert response_user.shape == (3, 4)
    assert response_item.shape == (4, 4)
    assert float(response_user[0].norm().detach()) == pytest.approx(1.0)
    torch.testing.assert_close(response_user[1:], torch.zeros_like(response_user[1:]))


def test_non_high_users_receive_exactly_zero_auxiliary_score():
    model = _model()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 3])
    components = model.candidate_score_components(users, items)

    assert components["response"][0].abs() > 0
    torch.testing.assert_close(
        components["response"][1:],
        torch.zeros_like(components["response"][1:]),
        atol=0,
        rtol=0,
    )


def test_rho_is_true_auxiliary_score_bound():
    model = _model(rho=0.05)
    users = torch.zeros(4, dtype=torch.long)
    items = torch.arange(4)
    response = model.candidate_score_components(users, items)["response"]

    assert response.abs().max() <= 0.05 + 1e-7


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def test_one_plain_bpr_updates_id_and_item_response_in_one_graph():
    model = _model()
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
    assert model.item_response.weight.grad.abs().sum() > 0


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


@pytest.mark.parametrize(
    "gate",
    [
        np.array([1.0, 0.5, 0.0], np.float32),
        np.ones(3, np.float32),
        np.zeros(3, np.float32),
    ],
)
def test_invalid_or_degenerate_high_clv_gate_is_rejected(gate):
    with pytest.raises(ValueError, match="high_clv_gate|고CLV"):
        HighCLVItemResponseLightGCN(
            n_users=3,
            n_items=4,
            high_clv_gate=gate,
            adj=_adj(),
            id_dim=6,
            response_dim=4,
        )
