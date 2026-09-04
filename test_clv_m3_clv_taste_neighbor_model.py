import numpy as np
import torch

from clv_m3_clv_conditioned_candidate_item_model import (
    build_binary_directional_blocks,
)
from clv_m3_clv_taste_neighbor_model import CLVTasteNeighborLightGCN


def _sparse(rows, cols, values, shape):
    return torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        size=shape,
    ).coalesce()


def _model(*, gamma=0.075, second_row_active=True):
    user_item, item_user = build_binary_directional_blocks(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 2]),
        2,
        3,
        torch.device("cpu"),
    )
    if second_row_active:
        relation = _sparse([0, 1], [1, 0], [1.0, 1.0], (2, 2))
    else:
        relation = _sparse([0], [1], [1.0], (2, 2))
    model = CLVTasteNeighborLightGCN(
        n_users=2,
        n_items=3,
        base_user_from_item=user_item,
        base_item_from_user=item_user,
        user_neighbor_operator=relation,
        gamma=gamma,
        dim=3,
        n_layers=2,
        pref_reg=0.0,
    )
    with torch.no_grad():
        model.E_u.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        model.E_i.weight.copy_(
            torch.tensor(
                [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]]
            )
        )
    return model


def test_zero_gamma_is_bitwise_exact_m1():
    model = _model(gamma=0.0)
    base_user, base_item = model.m1_embeddings()
    actual_user, actual_item, *_ = model.embeddings()

    assert torch.equal(actual_user, base_user)
    assert torch.equal(actual_item, base_item)


def test_user_without_neighbor_is_bitwise_exact_m1():
    model = _model(second_row_active=False)
    base_user, base_item = model.m1_embeddings()
    actual_user, actual_item, *_ = model.embeddings()

    assert torch.equal(actual_user[1], base_user[1])
    assert torch.equal(actual_item, base_item)


def test_only_user_layer_two_receives_the_fixed_neighbor_mixture():
    model = _model(gamma=0.075)
    parts = model.representation_parts()
    expected = (
        0.925 * parts["m1_user2"] + 0.075 * parts["neighbor_message"]
    )

    torch.testing.assert_close(parts["arm_user2"], expected)
    expected_user = (
        parts["user0"] + parts["user1"] + parts["arm_user2"]
    ) / 3.0
    actual_user, actual_item, *_ = model.embeddings()
    torch.testing.assert_close(actual_user, expected_user)
    assert torch.equal(actual_item, parts["m1_item"])


def test_plain_bpr_updates_both_id_embedding_tables_only():
    model = _model()
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([2, 0]),
        lam=0.0,
        weights=None,
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad is not None
    assert model.E_i.weight.grad is not None
    assert model.E_u.weight.grad.norm() > 0
    assert model.E_i.weight.grad.norm() > 0
    assert list(dict(model.named_parameters())) == ["E_u.weight", "E_i.weight"]
