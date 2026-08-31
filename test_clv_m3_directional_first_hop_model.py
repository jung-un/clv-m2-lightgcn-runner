import numpy as np
import torch

from clv_m3_directional_first_hop_model import DirectionalFirstHopLightGCN


def _sparse(values):
    # Edges: (u0,i0), (u0,i1), (u1,i1), (u1,i2)
    rows = torch.tensor([0, 0, 1, 1])
    cols = torch.tensor([0, 1, 1, 2])
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.tensor(values, dtype=torch.float32),
        size=(2, 3),
    ).coalesce()


def _base_values():
    # d_u=2 for both; d_i=(1,2,1)
    return np.array([1 / np.sqrt(2), 0.5, 0.5, 1 / np.sqrt(2)])


def _model(active_values=None):
    base = _sparse(_base_values())
    active = _sparse(_base_values() if active_values is None else active_values)
    model = DirectionalFirstHopLightGCN(
        n_users=2,
        n_items=3,
        base_user_from_item=base,
        base_item_from_user=base.transpose(0, 1),
        active_user_from_item=active,
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


def _ordinary_lightgcn(model):
    user_item = model.base_user_from_item.to_dense()
    item_user = model.base_item_from_user.to_dense()
    block = torch.cat(
        [
            torch.cat([torch.zeros(2, 2), user_item], dim=1),
            torch.cat([item_user, torch.zeros(3, 3)], dim=1),
        ],
        dim=0,
    )
    layer0 = torch.cat([model.E_u.weight, model.E_i.weight])
    layer1 = block @ layer0
    layer2 = block @ layer1
    final = (layer0 + layer1 + layer2) / 3
    return final[:2], final[2:]


def test_binary_active_operator_is_exact_two_layer_m1():
    model = _model()
    expected_user, expected_item = _ordinary_lightgcn(model)
    actual_user, actual_item, *_ = model.embeddings()
    torch.testing.assert_close(actual_user, expected_user)
    torch.testing.assert_close(actual_item, expected_item)


def test_active_graph_changes_only_final_user_first_hop_term():
    base = _base_values()
    # Preserve each user's row mass while redistributing its two coefficients.
    active = base.copy()
    active[0], active[1] = base[:2].sum() * 0.75, base[:2].sum() * 0.25
    active[2], active[3] = base[2:].sum() * 0.25, base[2:].sum() * 0.75
    model = _model(active)
    layers = model.layer_embeddings()

    torch.testing.assert_close(layers["item1_final"], layers["item1_m1"])
    torch.testing.assert_close(layers["user2_final"], layers["user2_m1"])
    torch.testing.assert_close(layers["item2_final"], layers["item2_m1"])
    assert not torch.allclose(layers["user1_final"], layers["user1_m1"])

    final_user, final_item, *_ = model.embeddings()
    expected_user = (
        layers["user0"] + layers["user1_final"] + layers["user2_m1"]
    ) / 3
    expected_item = (
        layers["item0"] + layers["item1_m1"] + layers["item2_m1"]
    ) / 3
    torch.testing.assert_close(final_user, expected_user)
    torch.testing.assert_close(final_item, expected_item)


def test_plain_bpr_updates_both_id_embedding_tables_in_one_graph():
    model = _model()
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([2, 0]),
        gate=torch.ones(2),
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

