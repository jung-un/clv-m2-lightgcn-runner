import numpy as np
import torch

from clv_m5_semantic_nv_m3_model import (
    M5SemanticNVIsolatedFirstHopLightGCN,
)
from clv_m5_semantic_nv_model import M5SemanticNVEconomicLightGCN


def _operators(active_values=None):
    rows = torch.tensor([0, 0, 1, 1])
    cols = torch.tensor([0, 1, 1, 2])
    base_values = torch.tensor(
        [1 / np.sqrt(2), 0.5, 0.5, 1 / np.sqrt(2)], dtype=torch.float32
    )
    active_values = (
        base_values
        if active_values is None
        else torch.tensor(active_values, dtype=torch.float32)
    )
    base_ui = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), base_values, size=(2, 3)
    ).coalesce()
    active_ui = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), active_values, size=(2, 3)
    ).coalesce()
    full_rows = torch.cat([rows, 2 + cols])
    full_cols = torch.cat([2 + cols, rows])
    full_values = torch.cat([base_values, base_values])
    full = torch.sparse_coo_tensor(
        torch.stack([full_rows, full_cols]), full_values, size=(5, 5)
    ).coalesce()
    return base_ui, base_ui.transpose(0, 1).coalesce(), active_ui, full


def _kwargs(adj):
    return {
        "n_users": 2,
        "n_items": 3,
        "user_q_n": np.array([0.25, 0.75]),
        "user_q_v_centered": np.array([-0.5, 0.5]),
        "user_centered_profile": np.array(
            [[0.2, -0.2], [-0.3, 0.3]], dtype=np.float32
        ),
        "user_economic_valid": np.array([True, True]),
        "item_price_centered": np.array([-1.0, 0.0, 1.0]),
        "item_centered_bin": np.array(
            [[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]], dtype=np.float32
        ),
        "item_economic_valid": np.array([True, True, True]),
        "adj": adj,
        "id_dim": 2,
        "rho": 0.15,
        "beta": 0.25,
        "n_layers": 2,
        "pref_reg": 0.0,
    }


def _isolated(active_values=None):
    base_ui, base_iu, active_ui, full = _operators(active_values)
    return M5SemanticNVIsolatedFirstHopLightGCN(
        base_user_from_item=base_ui,
        base_item_from_user=base_iu,
        active_user_from_item=active_ui,
        **_kwargs(full),
    )


def test_m3_off_is_exact_semantic_lineage_b_with_copied_state():
    *_, full = _operators()
    reference = M5SemanticNVEconomicLightGCN(**_kwargs(full))
    isolated = _isolated()
    isolated.load_state_dict(reference.state_dict(), strict=True)

    reference_user, reference_item = reference.propagated_embeddings()
    isolated_user, isolated_item = isolated.propagated_embeddings()
    torch.testing.assert_close(isolated_user, reference_user, atol=1e-6, rtol=0)
    torch.testing.assert_close(isolated_item, reference_item, atol=1e-6, rtol=0)


def test_active_m3_changes_only_user_layer_one_term():
    base = np.array([1 / np.sqrt(2), 0.5, 0.5, 1 / np.sqrt(2)])
    active = base.copy()
    active[:2] = base[:2].sum() * np.array([0.75, 0.25])
    active[2:] = base[2:].sum() * np.array([0.25, 0.75])
    model = _isolated(active)
    layers = model.layer_embeddings()

    assert not torch.allclose(layers["user1_active"], layers["user1_base"])
    user, item = model.id_embeddings()
    torch.testing.assert_close(
        user,
        (layers["user0"] + layers["user1_active"] + layers["user2_base"]) / 3,
    )
    torch.testing.assert_close(
        item,
        (layers["item0"] + layers["item1_base"] + layers["item2_base"]) / 3,
    )


def test_one_backward_connects_id_and_both_semantic_calibrations():
    model = _isolated()
    user, item = model.propagated_embeddings()
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[2, 1], [0, 1]])
    positive = (user[users] * item[positives]).sum(1)
    negative = (user[users, None] * item[negatives]).sum(2)
    loss = torch.nn.functional.softplus(negative - positive[:, None]).mean()
    loss.backward()

    assert model.E_u.weight.grad.norm() > 0
    assert model.E_i.weight.grad.norm() > 0
    assert model.value_scale_parameter.grad.abs() > 0
    assert model.profile_scale_parameter.grad.abs() > 0
