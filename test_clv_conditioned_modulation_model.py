import numpy as np
import pytest
import torch

from clv_conditioned_modulation_model import CLVConditionedModulationLightGCN
from clv_dual_axis_model import DualItemProfile


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


def _model(*, activity_valid=None, value_valid=None, layers=0):
    user_n = np.array(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.9], [0.4, 0.5, 0.6]], np.float32
    )
    user_v = np.array([[0.9, 0.8], [0.2, 0.1], [0.5, 0.6]], np.float32)
    item = DualItemProfile(
        activity=np.array(
            [[0.1, 0.2, 1.0], [0.8, 0.7, 1.0], [0.4, 0.6, 0.0], [0.2, 0.9, 1.0]],
            np.float32,
        ),
        value=np.array(
            [
                [0.8, 0.9, 0.7, 0.6],
                [0.2, 0.1, 0.3, 0.4],
                [0.5, 0.4, 0.6, 0.5],
                [0.9, 0.7, 0.8, 0.7],
            ],
            np.float32,
        ),
        valid_item=np.ones(4, bool),
        activity_names=("repeat_share", "gap", "mask"),
        value_names=("price", "category_price", "share", "mask"),
    )
    return CLVConditionedModulationLightGCN(
        n_users=3,
        n_items=4,
        user_activity=user_n,
        user_value=user_v,
        user_activity_valid=(
            np.ones(3, bool) if activity_valid is None else activity_valid
        ),
        user_value_valid=np.ones(3, bool) if value_valid is None else value_valid,
        item_profile=item,
        adj=_adj(),
        embedding_dim=6,
        modulation_rank=2,
        tau=0.10,
        n_layers=layers,
        pref_reg=1e-4,
    )


def _activate(module, value=0.2):
    module.output.weight.data.fill_(value)


def test_zero_output_projection_starts_exactly_at_id_embeddings():
    torch.manual_seed(42)
    model = _model()

    user, item = model.layer0_embeddings()

    torch.testing.assert_close(user, model.E_u.weight)
    torch.testing.assert_close(item, model.E_i.weight)
    assert user.shape == (3, 6)
    assert item.shape == (4, 6)


def test_nonzero_axis_modulation_changes_embeddings_without_adding_dimensions():
    model = _model()
    _activate(model.user_n)
    _activate(model.item_n)

    user, item = model.layer0_embeddings()

    assert user.shape[1] == item.shape[1] == 6
    assert not torch.allclose(user, model.E_u.weight)
    assert not torch.allclose(item, model.E_i.weight)
    ratio = user / model.E_u.weight
    assert torch.all(ratio <= 1.10 + 1e-6)
    assert torch.all(ratio >= 0.90 - 1e-6)


def test_invalid_user_axis_is_zero_after_the_modulator():
    model = _model(
        activity_valid=np.array([False, True, True]),
        value_valid=np.array([False, True, True]),
    )
    for module in (model.user_n, model.user_v):
        _activate(module)

    user, _ = model.layer0_embeddings()

    torch.testing.assert_close(user[0], model.E_u.weight[0])
    assert not torch.allclose(user[1], model.E_u.weight[1])


def test_one_plain_bpr_sends_gradient_to_id_and_all_modulators():
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

    assert diagnostics["bpr"] > 0
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    for module in (model.user_n, model.user_v, model.item_n, model.item_v):
        assert module.output.weight.grad.abs().sum() > 0


def test_loss_rejects_sample_weights_so_m4_cannot_leak_into_m2():
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


def test_modulation_diagnostics_separates_n_and_v_effects():
    model = _model()
    _activate(model.user_n)
    _activate(model.item_v, 0.3)

    diagnostics = model.modulation_diagnostics()

    assert diagnostics["tau"] == pytest.approx(0.10)
    assert diagnostics["user_n_abs_mean"] > 0
    assert diagnostics["user_v_abs_mean"] == 0
    assert diagnostics["item_n_abs_mean"] == 0
    assert diagnostics["item_v_abs_mean"] > 0
    assert 0 <= diagnostics["combined_saturation_share"] <= 1
