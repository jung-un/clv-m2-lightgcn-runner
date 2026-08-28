import numpy as np
import pandas as pd
import pytest
import torch

from clv_dynamic_clv_modulation_model import DynamicCLVModulationLightGCN
import lightgcn_clv_dynamic_multianchor as runner


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
    return DynamicCLVModulationLightGCN(
        n_users=2,
        n_items=3,
        adjacencies=[_adj(), _adj()],
        clv_conditions=[
            np.array([-1.0, 0.5], np.float32),
            np.array([0.75, -0.25], np.float32),
        ],
        context_names=["anchor_1", "anchor_2"],
        embedding_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_rho_zero_is_exactly_the_unconditioned_layer0_embedding():
    torch.manual_seed(42)
    model = _model(rho=0.0)
    model.set_context("anchor_2")

    user, item = model.layer0_embeddings()

    torch.testing.assert_close(user, model.E_u.weight, rtol=0, atol=0)
    torch.testing.assert_close(item, model.E_i.weight, rtol=0, atol=0)
    assert model.clv_dimension_weight.requires_grad is False


def test_one_shared_user_embedding_changes_with_the_historical_clv_context():
    model = _model()
    model.clv_dimension_weight.data.copy_(
        torch.tensor([0.8, -0.4, 0.2, -0.7])
    )
    original_norm = model.E_u.weight.norm(dim=1)

    model.set_context("anchor_1")
    first, first_item = model.layer0_embeddings()
    model.set_context("anchor_2")
    second, second_item = model.layer0_embeddings()

    assert not torch.allclose(first, second)
    torch.testing.assert_close(first.norm(dim=1), original_norm)
    torch.testing.assert_close(second.norm(dim=1), original_norm)
    torch.testing.assert_close(first_item, second_item)


def test_plain_bpr_jointly_updates_id_and_clv_conditioning_parameter():
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
    assert model.clv_dimension_weight.grad.abs().sum() > 0


def test_anchor_targets_keep_only_future_first_time_warm_pairs():
    frame = pd.DataFrame(
        {
            "u_idx": [0, 1, 1, 0, 0, 1],
            "i_idx": [0, 1, 0, 0, 1, 1],
            "t": [1, 2, 3, 6, 6, 6],
            "v": [10.0, 20.0, 5.0, 8.0, 9.0, 7.0],
            "b_raw": ["a", "b", "c", "d", "d", "e"],
        }
    )

    context = runner._build_training_context(
        frame,
        history_end=5,
        target_end=7,
        n_users=2,
        n_items=2,
    )

    assert list(zip(context["tr_u"], context["tr_i"])) == [(0, 1)]
    assert context["stats"]["new_item_target_pairs"] == 1
    assert context["stats"]["target_rows_before_new_pair_filter"] == 3


def test_historical_clv_is_recomputed_from_basket_frequency_and_value():
    history = pd.DataFrame(
        {
            "u_idx": [0, 1, 1],
            "i_idx": [0, 0, 1],
            "t": [1, 1, 2],
            "v": [5.0, 10.0, 30.0],
            "b_raw": ["a", "b", "c"],
        }
    )

    result = runner.historical_clv_condition(history, n_users=3)

    assert result["valid"].tolist() == [True, True, False]
    assert result["condition"][2] == 0.0
    assert result["condition"][1] > result["condition"][0]
    assert np.max(np.abs(result["condition"])) <= 1.0


def test_preflight_locks_m2_boundaries_and_development_split(tmp_path):
    cfg = runner.configure_dynamic_multianchor(out_dir=str(tmp_path))
    summary = runner.preflight_summary(cfg)

    assert summary["models"] == [
        "m1_multianchor_rho0",
        "m2_dynamic_clv",
    ]
    assert summary["historical_development_split"] == {
        "multi_anchor_target_start": 656,
        "multi_anchor_target_end": 683,
        "evaluation_start": 684,
        "evaluation_end": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"seed": 43}, "seed=42"),
        ({"rho": 0.1}, "rho=0.05"),
        ({"epochs": 50}, "epochs=100"),
        ({"anchor_count": 3}, "anchor_count=4"),
    ],
)
def test_unplanned_search_variants_are_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        runner.configure_dynamic_multianchor(
            out_dir=str(tmp_path), **override
        )
