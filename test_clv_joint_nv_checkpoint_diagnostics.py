import numpy as np
import torch

from clv_dual_axis_model import DualItemProfile
from clv_joint_nv_diagnostics import (
    JointNVBlockView,
    axis_distribution_diagnostics,
    block_score_diagnostics,
    evaluate_block_views,
    find_joint_checkpoint,
    load_joint_checkpoint,
)
from clv_joint_nv_model import JointNVLightGCN


def _model():
    n_users, n_items = 3, 4
    rows = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6]
    cols = [3, 4, 4, 5, 6, 0, 0, 0, 1, 1, 2, 2]
    adj = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.ones(len(rows)), (7, 7)
    ).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1)
    idx = adj.indices()
    adj = torch.sparse_coo_tensor(
        idx,
        adj.values() / torch.sqrt(degree[idx[0]] * degree[idx[1]]),
        adj.shape,
    ).coalesce()
    item_profile = DualItemProfile(
        activity=np.array([[1, 0], [0, 1], [1, 1], [0.5, 0.5]], np.float32),
        value=np.array([[1], [2], [3], [4]], np.float32),
        valid_item=np.ones(n_items, bool),
        activity_names=("repeat", "gap"),
        value_names=("mean_value",),
    )
    return JointNVLightGCN(
        n_users=n_users,
        n_items=n_items,
        user_activity=np.array([[1, 0], [0, 1], [1, 1]], np.float32),
        user_value=np.array([[1], [2], [3]], np.float32),
        item_profile=item_profile,
        q_n=np.array([0.2, 0.5, 0.8], np.float32),
        q_v=np.array([0.8, 0.5, 0.2], np.float32),
        adj=adj,
        id_dim=2,
        axis_dim=1,
        hidden_dim=3,
        n_layers=1,
        gate_shape="equal",
        gamma_init=0.2,
    )


def test_block_scores_sum_exactly_to_full_joint_score():
    model = _model().eval()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 2, 3])

    diagnostics = block_score_diagnostics(model, users, items)

    torch.testing.assert_close(
        diagnostics["full_scores"],
        diagnostics["id_scores"]
        + diagnostics["n_scores"]
        + diagnostics["v_scores"],
    )
    assert diagnostics["reconstruction_max_abs_error"] < 1e-6


def test_block_view_masks_unselected_dimensions_after_propagation():
    model = _model().eval()
    full_u, full_i = model.propagate()

    id_view = JointNVBlockView(model, ("id",))
    id_u, id_i, _, _ = id_view.embeddings()
    nv_view = JointNVBlockView(model, ("n", "v"))
    nv_u, nv_i, _, _ = nv_view.embeddings()

    torch.testing.assert_close(id_u @ id_i.T + nv_u @ nv_i.T, full_u @ full_i.T)
    assert torch.count_nonzero(id_u[:, 2:]) == 0
    assert torch.count_nonzero(nv_u[:, :2]) == 0


def test_axis_distribution_uses_midranks_and_reports_ties():
    diagnostics = axis_distribution_diagnostics(
        np.array([0.0, 0.0, 0.0, 1.0], np.float32),
        np.array([10.0, 20.0, 30.0, 40.0], np.float32),
        np.ones(4, bool),
    )

    assert diagnostics["q_n_unique"] == 2
    assert diagnostics["n_max_tie_share"] == 0.75
    assert diagnostics["n_zero_share"] == 0.75
    assert diagnostics["q_n_min"] < 0.5 < diagnostics["q_n_max"]


def test_checkpoint_loader_selects_latest_compatible_checkpoint(tmp_path):
    model = _model()
    older = tmp_path / "joint_nv_hm_s42_old.pt"
    latest = tmp_path / "joint_nv_hm_s42_latest.pt"
    payload = {
        "state": model.state_dict(),
        "config": {"dataset": "hm", "seed": 42},
        "input_hash": "same-input",
    }
    torch.save(payload, older)
    torch.save(payload, latest)
    older.touch()
    latest.touch()

    selected = find_joint_checkpoint(tmp_path, dataset="hm", seed=42)
    loaded = load_joint_checkpoint(
        _model(), selected, dataset="hm", seed=42, input_hash="same-input"
    )

    assert selected == latest
    assert loaded["config"]["dataset"] == "hm"


def test_checkpoint_loader_rejects_different_input_data(tmp_path):
    model = _model()
    path = tmp_path / "joint_nv_hm_s42_x.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "config": {"dataset": "hm", "seed": 42},
            "input_hash": "old-input",
        },
        path,
    )

    with np.testing.assert_raises_regex(RuntimeError, "input hash"):
        load_joint_checkpoint(
            _model(), path, dataset="hm", seed=42, input_hash="new-input"
        )


def test_evaluate_block_views_runs_full_and_all_required_masks():
    model = _model().eval()

    def evaluator(view):
        user, item, _, _ = view.embeddings()
        return {"mean_score": float((user @ item.T).mean())}

    rows = evaluate_block_views(model, evaluator)

    assert set(rows) == {"id_only", "n_only", "v_only", "id_n", "id_v", "full"}
    np.testing.assert_allclose(
        rows["full"]["mean_score"],
        rows["id_only"]["mean_score"]
        + rows["n_only"]["mean_score"]
        + rows["v_only"]["mean_score"],
        atol=1e-6,
    )
