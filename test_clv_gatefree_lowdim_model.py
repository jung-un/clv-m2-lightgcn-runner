import inspect

import numpy as np
import pytest
import torch

from clv_gatefree_lowdim_model import GateFreeLowDimNVLightGCN


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
    norm = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, norm, raw.shape).coalesce()


def _model(
    layers=1,
    axis_budget=0.1,
    training_axis_balance_delta=0.0,
    independent_item_axes=False,
    axis_keep_probability=1.0,
):
    return GateFreeLowDimNVLightGCN(
        n_users=3,
        n_items=4,
        user_activity=np.array(
            [[-1.0, 0.2], [0.0, 0.5], [1.0, 0.8]], np.float32
        ),
        user_value=np.array(
            [[1.0, 0.7], [0.0, 0.5], [-1.0, 0.3]], np.float32
        ),
        user_activity_valid=np.ones(3, bool),
        user_value_valid=np.ones(3, bool),
        q_n=np.array([0.1, 0.5, 0.9], np.float32),
        q_v=np.array([0.9, 0.5, 0.1], np.float32),
        adj=_adj(),
        id_dim=6,
        axis_dim=4,
        hidden_dim=5,
        n_layers=layers,
        axis_budget=axis_budget,
        training_axis_balance_delta=training_axis_balance_delta,
        independent_item_axes=independent_item_axes,
        axis_keep_probability=axis_keep_probability,
        pref_reg=1e-4,
    )


def test_layer0_keeps_id_and_adds_two_bounded_four_dimensional_axes():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 14)
    assert item.shape == (4, 14)
    assert torch.all(user[:, 6:].abs() <= np.sqrt(0.1) + 1e-7)
    assert torch.all(item[:, 6:].abs() <= np.sqrt(0.1) + 1e-7)
    torch.testing.assert_close(user[:, 6:10].mean(0), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(user[:, 10:14].mean(0), torch.zeros(4), atol=1e-6, rtol=0)


def test_model_has_no_item_economic_inputs_gate_or_learned_axis_weight():
    parameters = inspect.signature(GateFreeLowDimNVLightGCN).parameters
    model = _model(layers=0)

    assert "item_profile" not in parameters
    assert not hasattr(model, "gate_n")
    assert not hasattr(model, "gate_v")
    assert not hasattr(model, "sqrt_gamma_n")
    assert not hasattr(model, "sqrt_gamma_v")
    assert model.axis_budget == pytest.approx(0.1)


def test_point_zero_five_budget_scales_each_axis_by_square_root():
    model = _model(layers=0, axis_budget=0.05)
    user, item = model.layer0_embeddings()

    assert torch.all(user[:, 6:].abs() <= np.sqrt(0.05) + 1e-7)
    assert torch.all(item[:, 6:].abs() <= np.sqrt(0.05) + 1e-7)
    assert model.representation_diagnostics()["axis_budget"] == pytest.approx(0.05)


def test_one_plain_bpr_loss_trains_id_user_axes_and_item_responses():
    model = _model(layers=1)
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([3, 0]),
        None,
        0.0,
        None,
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.activity_user.net[0].weight.grad.abs().sum() > 0
    assert model.value_user.net[0].weight.grad.abs().sum() > 0
    assert model.activity_item.weight.grad.abs().sum() > 0
    assert model.value_item.weight.grad.abs().sum() > 0


def test_m4_sample_weights_are_rejected():
    model = _model()
    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([3]),
            None,
            0.0,
            torch.tensor([2.0]),
        )


def test_training_balance_uses_one_complementary_epsilon_per_triplet():
    model = _model(layers=1, training_axis_balance_delta=0.3)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([3, 0])
    model.train()

    torch.manual_seed(123)
    rng_state = torch.get_rng_state()
    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )

    torch.set_rng_state(rng_state)
    epsilon = (torch.rand(2) * 2.0 - 1.0) * 0.3
    user, item = model.propagate()
    id_end = model.id_dim
    activity_end = id_end + model.axis_dim

    def score(item_ids):
        selected_user = user[users]
        selected_item = item[item_ids]
        id_score = (
            selected_user[:, :id_end] * selected_item[:, :id_end]
        ).sum(1)
        activity_score = (
            selected_user[:, id_end:activity_end]
            * selected_item[:, id_end:activity_end]
        ).sum(1)
        value_score = (
            selected_user[:, activity_end:]
            * selected_item[:, activity_end:]
        ).sum(1)
        return (
            id_score
            + (1.0 + epsilon) * activity_score
            + (1.0 - epsilon) * value_score
        )

    expected_bpr = -torch.nn.functional.logsigmoid(
        score(positives) - score(negatives)
    ).mean()
    expected = expected_bpr + model.batch_l2(users, positives, negatives)

    torch.testing.assert_close(loss, expected)
    assert diagnostics["training_axis_balance_delta"] == pytest.approx(0.3)
    assert torch.all(1.0 + epsilon >= 0.7)
    assert torch.all(1.0 - epsilon >= 0.7)
    torch.testing.assert_close(
        (1.0 + epsilon) + (1.0 - epsilon), torch.full((2,), 2.0)
    )


def test_evaluation_embeddings_do_not_apply_training_perturbation():
    plain = _model(layers=1, training_axis_balance_delta=0.0)
    perturbed = _model(layers=1, training_axis_balance_delta=0.3)
    perturbed.load_state_dict(plain.state_dict())
    plain.eval()
    perturbed.eval()

    plain_user, plain_item, *_ = plain.embeddings()
    perturbed_user, perturbed_item, *_ = perturbed.embeddings()

    torch.testing.assert_close(perturbed_user, plain_user)
    torch.testing.assert_close(perturbed_item, plain_item)


def test_independent_item_axes_do_not_route_axis_gradients_into_id_item_table():
    model = _model(layers=0, independent_item_axes=True)
    _, item = model.layer0_embeddings()

    item[:, model.id_dim :].sum().backward()

    assert (
        model.E_i.weight.grad is None
        or model.E_i.weight.grad.abs().sum() == 0
    )
    assert model.activity_item_embedding.weight.grad.abs().sum() > 0
    assert model.value_item_embedding.weight.grad.abs().sum() > 0


def test_training_block_dropout_uses_independent_inverted_axis_masks():
    model = _model(
        layers=1,
        independent_item_axes=True,
        axis_keep_probability=0.5,
    )
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([3, 0])
    model.train()

    torch.manual_seed(123)
    rng_state = torch.get_rng_state()
    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )

    user, item = model.propagate()
    id_end = model.id_dim
    activity_end = id_end + model.axis_dim

    def components(item_ids):
        selected_user = user[users]
        selected_item = item[item_ids]
        return (
            (selected_user[:, :id_end] * selected_item[:, :id_end]).sum(1),
            (
                selected_user[:, id_end:activity_end]
                * selected_item[:, id_end:activity_end]
            ).sum(1),
            (
                selected_user[:, activity_end:]
                * selected_item[:, activity_end:]
            ).sum(1),
        )

    positive = components(positives)
    negative = components(negatives)
    torch.set_rng_state(rng_state)
    keep_probability = 0.5
    activity_multiplier = (
        torch.rand_like(positive[0]) < keep_probability
    ).float() / keep_probability
    value_multiplier = (
        torch.rand_like(positive[0]) < keep_probability
    ).float() / keep_probability
    positive_score = (
        positive[0]
        + activity_multiplier * positive[1]
        + value_multiplier * positive[2]
    )
    negative_score = (
        negative[0]
        + activity_multiplier * negative[1]
        + value_multiplier * negative[2]
    )
    expected = -torch.nn.functional.logsigmoid(
        positive_score - negative_score
    ).mean() + model.batch_l2(users, positives, negatives)

    torch.testing.assert_close(loss, expected)
    assert diagnostics["axis_keep_probability"] == pytest.approx(0.5)
    assert diagnostics["activity_axis_active_share"] == pytest.approx(
        float((activity_multiplier > 0).float().mean())
    )
    assert diagnostics["value_axis_active_share"] == pytest.approx(
        float((value_multiplier > 0).float().mean())
    )


def test_evaluation_loss_keeps_both_axes_active_with_block_dropout_configured():
    model = _model(
        layers=1,
        independent_item_axes=True,
        axis_keep_probability=0.5,
    )
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([3, 0])
    model.eval()

    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )
    user, item = model.propagate()
    positive_score = (user[users] * item[positives]).sum(1)
    negative_score = (user[users] * item[negatives]).sum(1)
    expected = -torch.nn.functional.logsigmoid(
        positive_score - negative_score
    ).mean() + model.batch_l2(users, positives, negatives)

    torch.testing.assert_close(loss, expected)
    assert diagnostics["activity_axis_active_share"] == pytest.approx(1.0)
    assert diagnostics["value_axis_active_share"] == pytest.approx(1.0)


def test_balance_perturbation_and_block_dropout_cannot_be_combined():
    with pytest.raises(ValueError, match="동시에"):
        _model(
            training_axis_balance_delta=0.3,
            axis_keep_probability=0.5,
        )
