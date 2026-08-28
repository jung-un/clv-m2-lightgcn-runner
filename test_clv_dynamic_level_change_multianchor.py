import numpy as np
import pandas as pd
import pytest
import torch

from clv_dynamic_clv_level_change_model import (
    DynamicCLVLevelChangeLightGCN,
)
import lightgcn_clv_dynamic_level_change_multianchor as runner


def _adj(n_users=2, n_items=3):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2)]
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
    normalized = values / torch.sqrt(
        degree[indices[0]] * degree[indices[1]]
    )
    return torch.sparse_coo_tensor(
        indices, normalized, raw.shape
    ).coalesce()


def _model(rho=0.05, layers=0):
    return DynamicCLVLevelChangeLightGCN(
        n_users=2,
        n_items=3,
        adjacencies=[_adj(), _adj()],
        level_conditions=[
            np.array([-0.8, 0.5], np.float32),
            np.array([0.6, -0.3], np.float32),
        ],
        change_conditions=[
            np.array([0.7, -0.4], np.float32),
            np.array([-0.5, 0.9], np.float32),
        ],
        context_names=["anchor_1", "anchor_2"],
        embedding_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_rolling_clv_uses_only_the_fixed_recent_window_and_past_rows():
    history = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 0],
            "i_idx": [0, 1, 0, 1, 2],
            "t": [3, 9, 5, 7, 11],
            "v": [10.0, 30.0, 40.0, 10.0, 999.0],
            "b_raw": ["a", "b", "c", "d", "future"],
        }
    )

    with_future = runner.rolling_clv_level_change(
        history,
        n_users=2,
        history_end=10,
        window_days=4,
        change_lag_days=4,
    )
    without_future = runner.rolling_clv_level_change(
        history[history["t"] <= 10],
        n_users=2,
        history_end=10,
        window_days=4,
        change_lag_days=4,
    )

    np.testing.assert_allclose(
        with_future["level_condition"], without_future["level_condition"]
    )
    np.testing.assert_allclose(
        with_future["change_condition"], without_future["change_condition"]
    )
    assert with_future["current_clv_proxy"].tolist() == [30.0, 10.0]
    assert with_future["previous_clv_proxy"].tolist() == [10.0, 40.0]
    assert with_future["change_condition"][0] > 0
    assert with_future["change_condition"][1] < 0


def test_rho_zero_is_exactly_the_shared_unconditioned_embedding():
    torch.manual_seed(42)
    model = _model(rho=0.0)
    model.set_context("anchor_2")

    user, item = model.layer0_embeddings()

    torch.testing.assert_close(user, model.E_u.weight, rtol=0, atol=0)
    torch.testing.assert_close(item, model.E_i.weight, rtol=0, atol=0)
    assert model.level_dimension_weight.requires_grad is False
    assert model.change_dimension_weight.requires_grad is False


def test_level_and_change_make_distinct_context_specific_adjustments():
    model = _model()
    model.level_dimension_weight.data.copy_(
        torch.tensor([0.8, -0.4, 0.2, -0.7])
    )
    model.change_dimension_weight.data.copy_(
        torch.tensor([-0.1, 0.9, -0.6, 0.3])
    )
    original_norm = model.E_u.weight.norm(dim=1)

    model.set_context("anchor_1")
    first, _ = model.layer0_embeddings()
    model.set_context("anchor_2")
    second, _ = model.layer0_embeddings()

    assert not torch.allclose(first, second)
    torch.testing.assert_close(first.norm(dim=1), original_norm)
    torch.testing.assert_close(second.norm(dim=1), original_norm)


def test_plain_bpr_updates_ids_and_both_clv_conditioning_directions():
    model = _model(layers=1)
    model.set_context("anchor_1")

    loss, _ = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([2, 0]),
    )
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.level_dimension_weight.grad.abs().sum() > 0
    assert model.change_dimension_weight.grad.abs().sum() > 0


def test_preflight_locks_wide_anchors_recent_window_and_development_split(tmp_path):
    cfg = runner.configure_dynamic_level_change(out_dir=str(tmp_path))
    summary = runner.preflight_summary(cfg)

    assert summary["models"] == [
        "m1_wide_multianchor_rho0",
        "m2_dynamic_clv_level_change",
    ]
    assert summary["historical_development_split"] == {
        "anchor_history_ends": [480, 508, 536, 564, 592, 620, 648, 676],
        "anchor_target_horizon_days": 7,
        "evaluation_start": 684,
        "evaluation_end": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["historical_clv_proxy"]["rolling_window_days"] == 90
    assert summary["historical_clv_proxy"]["change_lag_days"] == 28
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"seed": 43}, "seed=42"),
        ({"rho": 0.1}, "rho=0.05"),
        ({"anchor_count": 4}, "anchor_count=8"),
        ({"anchor_spacing_days": 7}, "anchor_spacing_days=28"),
        ({"rolling_window_days": 30}, "rolling_window_days=90"),
    ],
)
def test_unplanned_search_variants_are_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        runner.configure_dynamic_level_change(
            out_dir=str(tmp_path), **override
        )
