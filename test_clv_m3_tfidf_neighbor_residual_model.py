import numpy as np
import torch

from clv_m3_clv_conditioned_candidate_item_model import (
    build_binary_directional_blocks,
)
from clv_m3_tfidf_neighbor_residual_model import (
    TFIDFNeighborResidualLightGCN,
)


def _sparse(rows, cols, values, shape):
    return torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        size=shape,
    ).coalesce()


def _model(*, gate=(1.0, 0.5), rho=0.075, active=True):
    user_item, item_user = build_binary_directional_blocks(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 2]),
        2,
        3,
        torch.device("cpu"),
    )
    relation = (
        _sparse([0, 1], [1, 0], [1.0, 1.0], (2, 2))
        if active
        else _sparse([], [], [], (2, 2))
    )
    model = TFIDFNeighborResidualLightGCN(
        n_users=2,
        n_items=3,
        base_user_from_item=user_item,
        base_item_from_user=item_user,
        user_neighbor_operator=relation,
        gate=torch.tensor(gate),
        rho=rho,
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


def test_zero_rho_is_bitwise_exact_m1():
    model = _model(rho=0.0)
    base_user, base_item = model.m1_embeddings()
    actual_user, actual_item, *_ = model.embeddings()

    assert torch.equal(actual_user, base_user)
    assert torch.equal(actual_item, base_item)


def test_invalid_neighbor_row_is_bitwise_exact_m1():
    model = _model(active=False)
    base_user, base_item = model.m1_embeddings()
    actual_user, actual_item, *_ = model.embeddings()

    assert torch.equal(actual_user, base_user)
    assert torch.equal(actual_item, base_item)


def test_residual_is_orthogonal_and_natural_size_is_preserved():
    model = _model()
    parts = model.representation_parts()
    dot = (parts["residual"] * parts["m1_user"]).sum(dim=1)

    torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-6, rtol=0)
    assert torch.all(parts["eta"] >= 0)
    assert torch.all(parts["eta"] <= 1 + 1e-6)
    expected_norm = parts["m1_user"].norm(dim=1) * parts["eta"]
    torch.testing.assert_close(
        parts["scaled_residual"].norm(dim=1), expected_norm, atol=1e-6, rtol=1e-6
    )


def test_final_user_norm_is_preserved_and_items_are_unchanged():
    model = _model()
    base_user, base_item = model.m1_embeddings()
    actual_user, actual_item, *_ = model.embeddings()

    torch.testing.assert_close(
        actual_user.norm(dim=1), base_user.norm(dim=1), atol=1e-6, rtol=1e-6
    )
    assert torch.equal(actual_item, base_item)


def test_plain_bpr_updates_both_embedding_tables_in_one_loop():
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

