import json
from pathlib import Path

import numpy as np
import torch

from clv_m5_semantic_nv_model import M5SemanticNVEconomicLightGCN
import lightgcn_clv_m5_semantic_nv_single_screen as runner


def _adj(n_users=2, n_items=3):
    edges = [(0, 0), (0, 1), (1, 2)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    with torch.sparse.check_sparse_tensor_invariants(False):
        raw = torch.sparse_coo_tensor(
            indices, values, (n_users + n_items,) * 2
        ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model(*, rho=0.15, beta=0.25, layers=1):
    torch.manual_seed(7)
    return M5SemanticNVEconomicLightGCN(
        n_users=2,
        n_items=3,
        user_q_n=np.array([0.25, 0.8], dtype=np.float32),
        user_q_v_centered=np.array([-0.6, 0.7], dtype=np.float32),
        user_centered_profile=np.array(
            [[0.1, -0.1, 0.0, 0.0], [-0.2, 0.0, 0.1, 0.1]],
            dtype=np.float32,
        ),
        user_economic_valid=np.ones(2, dtype=bool),
        item_price_centered=np.array([-0.8, 0.0, 0.9], dtype=np.float32),
        item_centered_bin=np.array(
            [[0.75, -0.25, -0.25, -0.25],
             [-0.25, 0.75, -0.25, -0.25],
             [-0.25, -0.25, -0.25, 0.75]],
            dtype=np.float32,
        ),
        item_economic_valid=np.ones(3, dtype=bool),
        adj=_adj(),
        id_dim=4,
        rho=rho,
        beta=beta,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_semantic_axes_keep_fixed_nv_and_price_meaning():
    model = _model(layers=0)
    user, item = model.economic_coordinates()

    expected_user = torch.tensor(
        [
            [-0.3, 0.0216506, -0.0216506, 0.0, 0.0],
            [0.35, -0.1385641, 0.0, 0.0692820, 0.0692820],
        ]
    )
    expected_item = torch.tensor(
        [
            [-0.4, 0.6495191, -0.2165064, -0.2165064, -0.2165064],
            [0.0, -0.2165064, 0.6495191, -0.2165064, -0.2165064],
            [0.45, -0.2165064, -0.2165064, -0.2165064, 0.6495191],
        ]
    )
    torch.testing.assert_close(user, expected_user, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(item, expected_item, atol=1e-6, rtol=1e-5)


def test_only_id_block_is_graph_propagated():
    model = _model(layers=2)
    user, item = model.propagated_embeddings()
    id_user, id_item = model.id_embeddings()
    economic_user, economic_item = model.economic_coordinates()

    torch.testing.assert_close(user[:, :4], id_user)
    torch.testing.assert_close(item[:, :4], id_item)
    torch.testing.assert_close(
        user[:, 4:], np.sqrt(0.15) * economic_user
    )
    torch.testing.assert_close(
        item[:, 4:], np.sqrt(0.15) * economic_item
    )


def test_same_loss_updates_id_and_bounded_semantic_calibrations():
    model = _model(layers=1)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[1], [0]])
    user, item = model.propagated_embeddings()
    loss = torch.nn.functional.softplus(
        (user[users, None, :] * item[negatives]).sum(2)
        - (user[users] * item[positives]).sum(1)[:, None]
    ).mean()
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.value_scale_parameter.grad.abs().sum() > 0
    assert model.profile_scale_parameter.grad.abs().sum() > 0
    assert 0.75 <= float(model.value_scale().detach()) <= 1.25
    assert 0.75 <= float(model.profile_scale().detach()) <= 1.25


def test_bounded_calibration_is_linear_in_score_not_squared():
    model = _model(layers=0)
    with torch.no_grad():
        model.value_scale_parameter.fill_(0.8)
        model.profile_scale_parameter.fill_(-0.6)
    user, item = model.economic_coordinates()

    value_score = user[1, 0] * item[2, 0]
    unscaled_value_score = 0.5 * 0.7 * 0.5 * 0.9
    profile_score = (user[1, 1:] * item[2, 1:]).sum()
    unscaled_profile_score = 0.75 * 0.8 * (
        torch.tensor([-0.2, 0.0, 0.1, 0.1])
        * torch.tensor([-0.25, -0.25, -0.25, 0.75])
    ).sum()

    torch.testing.assert_close(
        value_score / unscaled_value_score, model.value_scale()
    )
    torch.testing.assert_close(
        profile_score / unscaled_profile_score, model.profile_scale()
    )


def test_single_screen_has_only_observed_m5_arm(tmp_path):
    cfg = runner.configure_semantic_nv_single_screen(
        out_dir=str(tmp_path / "results")
    )
    summary = runner.preflight_summary(cfg)
    specs = runner.arm_specifications({}, cfg)

    assert summary["split"] == "historical_development_days_684_690"
    assert summary["trained_models"] == [runner.M5_MODEL_ID]
    assert summary["controls_trained"] is False
    assert summary["m2"]["economic_dim"] == 5
    assert summary["m2"]["economic_graph_propagation"] is False
    assert len(specs) == 1
    assert specs[0]["model_id"] == runner.M5_MODEL_ID
    assert specs[0]["weighted"] is True
    assert specs[0]["assignment_name"] == "observed"


def test_colab_runs_only_the_single_semantic_m5_arm():
    notebook_path = Path(
        "clv_m5_semantic_nv_personalized_positive_dunnhumby_single_seed_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "run_semantic_nv_single_screen(cfg)" in source
    assert "controls_trained" in source
    assert "historical_development_days_684_690" in source
    assert "run_m5_nv_economic_positive_test" not in source
