import numpy as np
import pytest
import torch

from clv_m5_economic_positive_weight_model import weighted_multi_negative_bpr
from clv_scaled_value_basis_gnn_model import CLVScaledValueBasisGNN
import gnn_clv_m1_m2_m4_m5_factorial_screen as runner


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
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model(backbone, m2_active):
    torch.manual_seed(7)
    return CLVScaledValueBasisGNN(
        backbone=backbone,
        m2_active=m2_active,
        n_users=3,
        n_items=4,
        user_q_v=np.array([0.1, 0.5, 0.9], np.float32),
        user_q_c=np.array([0.8, 0.5, 0.2], np.float32),
        user_clv_valid=np.ones(3, bool),
        item_price_percentile=np.array([0.0, 0.3, 0.7, 1.0], np.float32),
        item_price_valid=np.ones(4, bool),
        adj=_adj(),
        base_id_dim=6,
        economic_dim=3,
        rho=0.05,
        n_layers=2,
        pref_reg=1e-4,
    )


@pytest.mark.parametrize("backbone", ["ngcf", "gat", "graphsage"])
def test_four_arm_capacity_match_uses_9_total_coordinates(backbone):
    m1 = _model(backbone, False)
    m2 = _model(backbone, True)

    assert m1.trainable_id_dim == 9
    assert m2.trainable_id_dim == 6
    assert m1.layer0_embeddings()[0].shape == (3, 9)
    assert m2.layer0_embeddings()[0].shape == (3, 9)
    assert m2.layer0_embeddings(m2_active=False)[0].shape == (3, 9)
    if backbone == "ngcf":
        assert m1.propagated_embeddings()[0].shape == (3, 27)
        assert m2.propagated_embeddings()[0].shape == (3, 27)
    else:
        assert m1.propagated_embeddings()[0].shape == (3, 9)
        assert m2.propagated_embeddings()[0].shape == (3, 9)


@pytest.mark.parametrize("backbone", ["ngcf", "gat", "graphsage"])
def test_m2_coordinates_change_scores_and_receive_recommendation_gradient(backbone):
    model = _model(backbone, True)
    users = torch.tensor([0, 1, 2])
    positives = torch.tensor([0, 1, 3])
    negatives = torch.tensor([[2, 3], [0, 3], [0, 1]])

    full_user, full_item = model.propagated_embeddings()
    off_user, off_item = model.propagated_embeddings(m2_active=False)
    assert not torch.allclose(full_user, off_user)
    assert not torch.allclose(full_item, off_item)

    positive_scores = (full_user[users] * full_item[positives]).sum(dim=1)
    negative_scores = (
        full_user[users, None, :] * full_item[negatives]
    ).sum(dim=2)
    bpr, _ = weighted_multi_negative_bpr(
        positive_scores, negative_scores, torch.ones_like(positive_scores)
    )
    loss = bpr + model.sampled_l2(users, positives, negatives)
    loss.backward()
    gradients = model.training_gradient_diagnostics()

    assert gradients["id_user_gradient_norm"] > 0
    assert gradients["id_item_gradient_norm"] > 0
    assert gradients["backbone_layer0_gradient_norm"] > 0
    assert gradients["auxiliary_input_column_gradient_norm"] > 0


def test_preflight_fixes_development_protocol_and_four_factorial_arms():
    for backbone in ("ngcf", "gat", "graphsage"):
        cfg = runner.configure_gnn_clv_factorial_screen(
            backbone, out_dir=f"/tmp/{backbone}-factorial"
        )
        summary = runner.preflight_summary(cfg)
        ids = runner.model_ids(backbone)

        assert summary["trained_models"] == list(ids.values())
        assert summary["reused_models"] == []
        assert summary["split"] == "historical_development_days_684_690"
        assert summary["fixed"]["new_item_task"] is True
        assert summary["fixed"]["train_pairs_excluded_from_evaluation"] is True
        assert summary["fixed"]["min_item_interactions"] == 1
        assert summary["fixed"]["graph"] == "binary"
        assert summary["fixed"]["negative_sampling"] == "uniform"
        assert summary["fixed"]["final_test_constructed"] is False
        assert summary["fixed"]["holdout_constructed"] is False
        assert summary["fixed"]["one_training_loop_and_optimizer_per_arm"] is True
        assert summary["m2"]["same_recommendation_gradient"] is True
        assert summary["m2"]["external_reranking"] is False


def test_arm_specs_form_exact_m2_by_m4_factorial():
    prepared = {"q_c": np.array([0.5]), "user_bin_fit": np.ones((1, 4))}
    cfg = runner.configure_gnn_clv_factorial_screen(
        "ngcf", out_dir="/tmp/ngcf-factorial"
    )
    specs = runner.arm_specifications(prepared, cfg)

    assert [(spec["m2_active"], spec["weighted"]) for spec in specs] == [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    assert all(spec["assignment"] is prepared for spec in specs)


def test_interaction_is_m5_minus_m4_less_m2_minus_m1():
    ids = runner.model_ids("gat")
    rows = {
        ids["m1"]: {metric: 1.0 for metric in runner.CORE_METRICS},
        ids["m2"]: {metric: 1.1 for metric in runner.CORE_METRICS},
        ids["m4"]: {metric: 1.2 for metric in runner.CORE_METRICS},
        ids["m5"]: {metric: 1.5 for metric in runner.CORE_METRICS},
    }
    interaction = runner.interaction_rows(rows, ids)

    np.testing.assert_allclose(interaction["interaction"], 0.2)
