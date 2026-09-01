import numpy as np
import torch

from clv_m3_clv_conditioned_category_transition_model import (
    CLVCategoryTransitionLightGCN,
    build_binary_directional_blocks,
)


def _sparse(rows, cols, values, shape):
    return torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        size=shape,
    ).coalesce()


def _model(active=True):
    users = np.array([0, 0, 1, 1])
    items = np.array([0, 1, 1, 2])
    user_item, item_user = build_binary_directional_blocks(
        users, items, 2, 3, torch.device("cpu")
    )
    user_category = (
        _sparse([0, 1], [0, 1], [1.0, 1.0], (2, 2))
        if active
        else _sparse([], [], [], (2, 2))
    )
    category_item = _sparse(
        [0, 0, 1], [0, 1, 2], [0.5, 0.5, 1.0], (2, 3)
    )
    model = CLVCategoryTransitionLightGCN(
        n_users=2,
        n_items=3,
        n_cat=2,
        base_user_from_item=user_item,
        base_item_from_user=item_user,
        user_target_category=user_category,
        category_item=category_item,
        gamma=0.075,
        dim=2,
        n_layers=2,
        pref_reg=0.0,
    )
    with torch.no_grad():
        model.E_u.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.E_i.weight.copy_(
            torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        )
    return model


def test_zero_auxiliary_relation_is_exact_binary_m1():
    model = _model(active=False)
    base_user, base_item, message = model.representation_parts()
    actual_user, actual_item, *_ = model.embeddings()
    torch.testing.assert_close(message, torch.zeros_like(message))
    torch.testing.assert_close(actual_user, base_user)
    torch.testing.assert_close(actual_item, base_item)


def test_category_transition_changes_only_user_final_representation():
    model = _model(active=True)
    base_user, base_item, message = model.representation_parts()
    actual_user, actual_item, *_ = model.embeddings()
    torch.testing.assert_close(actual_user, base_user + 0.075 * message)
    torch.testing.assert_close(actual_item, base_item)
    assert not torch.allclose(actual_user, base_user)


def test_plain_bpr_jointly_updates_both_embedding_tables():
    model = _model(active=True)
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
