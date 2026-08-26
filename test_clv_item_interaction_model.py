import numpy as np
import torch

from clv_item_interaction_model import CLVItemInteractionLightGCN


def _adj():
    indices = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
    values = torch.ones(4)
    return torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()


def _model(clv_coordinate=(-0.5, 0.5)):
    torch.manual_seed(7)
    return CLVItemInteractionLightGCN(
        n_users=2,
        n_items=2,
        clv_coordinate=np.asarray(clv_coordinate, dtype=np.float32),
        adj=_adj(),
        embedding_dim=4,
        n_layers=1,
        pref_reg=1e-3,
    )


def test_zero_initialisation_starts_exactly_from_lightgcn_dot_product():
    model = _model()
    user, item, _, _ = model.embeddings()
    propagated_user, propagated_item = model.propagate()

    assert torch.count_nonzero(model.item_clv.weight) == 0
    torch.testing.assert_close(user[:, :-1], propagated_user)
    torch.testing.assert_close(item[:, :-1], propagated_item)
    torch.testing.assert_close(user[:, -1], model.clv_coordinate)
    torch.testing.assert_close(item[:, -1], torch.zeros(2))


def test_score_is_lightgcn_plus_clv_coordinate_times_item_coefficient():
    model = _model()
    with torch.no_grad():
        model.item_clv.weight[:, 0] = torch.tensor([2.0, -3.0])
    user, item, _, _ = model.embeddings()
    base_user, base_item = model.propagate()
    score = user @ item.T
    expected = base_user @ base_item.T + model.clv_coordinate[:, None] * torch.tensor([2.0, -3.0])
    torch.testing.assert_close(score, expected)


def test_plain_bpr_gradient_updates_item_clv_without_sample_weights():
    model = _model()
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    negatives = torch.tensor([1, 0])
    loss, diagnostics = model.bpr_loss(users, positives, negatives, lam=0.0)
    loss.backward()

    assert model.item_clv.weight.grad is not None
    assert torch.count_nonzero(model.item_clv.weight.grad) > 0
    assert set(diagnostics) == {"bpr", "p_correct"}


def test_m4_weights_and_external_lambda_are_rejected():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([1])
    try:
        model.bpr_loss(users, positives, negatives, lam=1.0)
        raise AssertionError("external lambda must fail")
    except ValueError:
        pass
    try:
        model.bpr_loss(users, positives, negatives, w=torch.ones(1))
        raise AssertionError("M4 weights must fail")
    except ValueError:
        pass
