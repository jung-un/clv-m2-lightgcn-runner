import numpy as np
import torch

from clv_dual_axis_model import DualItemProfile
from clv_postprop_gate_model import PostPropagationGatedJointNVLightGCN


def _adj():
    n_users, n_items = 3, 4
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows))
    raw = torch.sparse_coo_tensor(
        indices, values, (n_users + n_items,) * 2
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model(activity_valid=None):
    item = DualItemProfile(
        activity=np.array(
            [[0.1, 0.2, 1], [0.8, 0.7, 1], [0.4, 0.6, 0], [0.2, 0.9, 1]],
            np.float32,
        ),
        value=np.array(
            [[0.8, 0.9], [0.2, 0.1], [0.5, 0.4], [0.9, 0.7]],
            np.float32,
        ),
        valid_item=np.ones(4, bool),
        activity_names=("a", "b", "c"),
        value_names=("d", "e"),
    )
    return PostPropagationGatedJointNVLightGCN(
        n_users=3,
        n_items=4,
        user_activity=np.array(
            [[0.1, 0.2], [0.8, 0.7], [0.4, 0.5]], np.float32
        ),
        user_value=np.array(
            [[0.9, 0.8], [0.2, 0.1], [0.5, 0.6]], np.float32
        ),
        user_activity_valid=(
            np.ones(3, bool) if activity_valid is None else activity_valid
        ),
        user_value_valid=np.ones(3, bool),
        item_profile=item,
        q_n=np.array([0.1, 0.9, 0.5], np.float32),
        q_v=np.array([0.9, 0.1, 0.5], np.float32),
        adj=_adj(),
        id_dim=6,
        axis_dim=3,
        hidden_dim=5,
        n_layers=1,
        variant="joint_nv",
        gate_shape="axis_positive",
        pref_reg=0.0,
        preference_preserving=True,
    )


def test_postprop_model_has_no_learned_global_axis_weight():
    model = _model()
    names = [name for name, _ in model.named_parameters()]

    assert not any("gamma" in name for name in names)
    assert model.activity_axis_weight is None
    assert model.transaction_value_axis_weight is None


def test_axis_gates_are_applied_after_shared_propagation_and_norms_are_bounded():
    model = _model()
    layer0_user, _ = model.layer0_embeddings()
    final_user, final_item = model.propagate()

    # User gates are absent at layer 0, then appear in final axis norms.
    torch.testing.assert_close(
        layer0_user[:, 6:9].norm(dim=1), torch.ones(3), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        final_user[:, 6:9].norm(dim=1), model.gate_n, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        final_user[:, 9:12].norm(dim=1), model.gate_v, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        final_item[:, 6:9].norm(dim=1), torch.ones(4), rtol=1e-5, atol=1e-5
    )


def test_invalid_user_axis_stays_zero_after_propagation():
    model = _model(activity_valid=np.array([True, False, True]))
    final_user, _ = model.propagate()

    assert torch.count_nonzero(final_user[1, 6:9]) == 0
    assert torch.count_nonzero(final_user[0, 6:9]) > 0


def test_one_bpr_backward_updates_id_and_both_axis_encoders():
    model = _model()
    loss, _ = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([3, 0]),
        None,
        0.0,
        None,
    )
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.activity_user.net[0].weight.grad.abs().sum() > 0
    assert model.activity_item.net[0].weight.grad.abs().sum() > 0
    assert model.value_user.net[0].weight.grad.abs().sum() > 0
    assert model.value_item.net[0].weight.grad.abs().sum() > 0
