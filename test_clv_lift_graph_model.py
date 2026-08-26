import numpy as np
import torch

from clv_lift_graph_model import CLVLiftGraphLightGCN


def _model(edge_signal=(0.0, 0.0, 0.0)):
    return CLVLiftGraphLightGCN(
        n_users=2,
        n_items=2,
        edge_users=np.array([0, 0, 1]),
        edge_items=np.array([0, 1, 1]),
        edge_signal=np.asarray(edge_signal, dtype=np.float32),
        embedding_dim=4,
        n_layers=1,
        pref_reg=1e-3,
        alpha_init=0.1,
    )


def test_zero_lift_signal_reproduces_binary_lightgcn_normalisation():
    model = _model()
    dense = model.weighted_adjacency().to_dense()
    root_half = 1.0 / np.sqrt(2.0)
    expected = torch.tensor(
        [
            [0.0, 0.0, root_half, 0.5],
            [0.0, 0.0, 0.0, root_half],
            [root_half, 0.0, 0.0, 0.0],
            [0.5, root_half, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(dense, expected)


def test_clv_changes_only_graph_values_not_embedding_dimensions():
    model = _model((0.5, -0.5, 0.25))
    user, item, _, _ = model.embeddings()

    assert user.shape == (2, 4)
    assert item.shape == (2, 4)
    assert model.weighted_adjacency().values().std() > 0


def test_plain_bpr_updates_positive_graph_strength_parameter():
    model = _model((0.8, -0.6, 0.3))
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    negatives = torch.tensor([1, 0])

    loss, _ = model.bpr_loss(users, positives, negatives, None, 0.0, None)
    loss.backward()

    assert model.raw_alpha.grad is not None
    assert float(model.raw_alpha.grad.abs()) > 0
    assert float(model.alpha().detach()) > 0


def test_graph_strength_cannot_expand_clipped_lift_beyond_declared_range():
    model = _model((np.log(3.0), -np.log(3.0), 0.0))
    with torch.no_grad():
        model.raw_alpha.fill_(100.0)

    assert float(model.alpha().detach()) <= 1.0
    raw_weight = torch.exp(model.alpha() * model.edge_signal)
    assert float(raw_weight.max().detach()) <= 3.0 + 1e-6
    assert float(raw_weight.min().detach()) >= (1.0 / 3.0) - 1e-6


def test_sparse_adjacency_values_do_not_request_a_dense_adjacency_gradient():
    model = _model((0.8, -0.6, 0.3))

    adjacency = model.weighted_adjacency()

    assert adjacency.layout == torch.sparse_coo
    assert adjacency.values().requires_grad is False


def test_manual_scalar_alpha_gradient_matches_finite_difference():
    model = _model((0.8, -0.6, 0.3))
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    negatives = torch.tensor([1, 0])
    loss, _ = model.bpr_loss(users, positives, negatives, None, 0.0, None)
    loss.backward()
    analytic = float(model.raw_alpha.grad)

    original = float(model.raw_alpha.detach())
    epsilon = 1e-3
    losses = []
    for value in (original + epsilon, original - epsilon):
        with torch.no_grad():
            model.raw_alpha.fill_(value)
        perturbed, _ = model.bpr_loss(
            users, positives, negatives, None, 0.0, None
        )
        losses.append(float(perturbed.detach()))
    numerical = (losses[0] - losses[1]) / (2.0 * epsilon)

    assert np.isclose(analytic, numerical, rtol=2e-2, atol=2e-4)


def test_m4_sample_weights_and_external_score_lambda_are_rejected():
    model = _model((0.8, -0.6, 0.3))
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([1])

    try:
        model.bpr_loss(users, positives, negatives, None, 1.0, None)
        raise AssertionError("external score lambda must fail")
    except ValueError:
        pass
    try:
        model.bpr_loss(users, positives, negatives, None, 0.0, torch.ones(1))
        raise AssertionError("M4 weights must fail")
    except ValueError:
        pass
