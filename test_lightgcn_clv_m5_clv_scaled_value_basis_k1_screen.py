import numpy as np
import pytest
import torch

import lightgcn_clv_m5_clv_scaled_value_basis_k1_screen as screen


def _adj(n_users=2, n_items=3):
    edges = [(0, 0), (0, 1), (1, 2)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.full((len(rows),), 0.5)
    return torch.sparse_coo_tensor(
        indices, values, (n_users + n_items,) * 2, check_invariants=False
    ).coalesce()


def _model(*, rho, propagate):
    from clv_m5_n_conditioned_value_basis_model import (
        M5NConditionedValueBasisLightGCN,
    )

    torch.manual_seed(3)
    return M5NConditionedValueBasisLightGCN(
        n_users=2,
        n_items=3,
        user_q_n=np.array([0.2, 0.9], dtype=np.float32),
        user_q_v=np.array([0.15, 0.8], dtype=np.float32),
        user_q_c=np.array([0.1, 0.9], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_price_percentile=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        item_price_valid=np.array([True, True, True]),
        adj=_adj(),
        id_dim=4,
        rho=rho,
        n_layers=2,
        economic_propagation=propagate,
    )


def test_unpropagated_value_block_is_appended_unchanged_after_id_propagation():
    model = _model(rho=0.25, propagate=False)

    user, item = model.propagated_embeddings()
    user_id, item_id = model.id_embeddings()
    user_value, item_value = model.economic_coordinates()

    torch.testing.assert_close(user[:, :4], user_id)
    torch.testing.assert_close(user[:, 4:], 0.5 * user_value)
    torch.testing.assert_close(item[:, 4:], 0.5 * item_value)
    assert model.representation_diagnostics()["economic_graph_propagation"] is False


def test_rho_zero_arm_scores_exactly_like_plain_lightgcn():
    model = _model(rho=0.0, propagate=False)

    user, item = model.propagated_embeddings()
    user_id, item_id = model.id_embeddings()

    torch.testing.assert_close(user @ item.T, user_id @ item_id.T)


def test_config_fixes_single_negative_and_rho():
    cfg = screen.configure_value_basis_k1_screen(
        out_dir="/tmp/out", baseline_result_dir="/tmp/base"
    )
    assert cfg.negative_count == 1
    assert cfg.rho == 0.25
    with pytest.raises(ValueError, match="negative_count"):
        screen.configure_value_basis_k1_screen(
            out_dir="/tmp/out", baseline_result_dir="/tmp/base", negative_count=5
        )


def test_four_arms_share_one_run_and_split_rho_and_weight_factorially():
    cfg = screen.configure_value_basis_k1_screen(
        out_dir="/tmp/out", baseline_result_dir="/tmp/base"
    )
    specs = screen.arm_specifications({"m2_actual": {}}, cfg)

    assert [spec["model_id"] for spec in specs] == list(screen.MODEL_IDS)
    assert [(spec["rho"] > 0, spec["weighted"]) for spec in specs] == [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    summary = screen.preflight_summary(cfg)
    assert summary["reused_models"] == []
    assert summary["loss"]["negative_count"] == 1
    assert summary["m2"]["economic_graph_propagation"] is False


def test_band_error_pairs_count_same_and_cross_band_pairs():
    counts = screen.band_error_pairs(
        users=np.array([7]),
        topk=np.array([[1, 2]]),
        truth={7: np.array([2, 3, 4])},
        item_bin=np.array([0, 0, 1, 0, 1]),
        k=2,
    )
    # missed truths 3 (bin 0), 4 (bin 1); false positive 1 (bin 0)
    assert counts == {"same_band_error_pairs": 1, "cross_band_error_pairs": 1}


def _metrics(econ, recall=0.02, ndcg=0.02):
    return {
        "recall@10": recall,
        "ndcg@10": ndcg,
        "recall@20": 0.03,
        "ndcg@20": 0.03,
        "recall@50": 0.05,
        "ndcg@50": 0.04,
        "price_purchase_amount_weighted_hit@10": econ,
        "vndcg@10": econ / 10,
    }


def _reading(m5_econ, change_share=0.3, m5_recall=0.02):
    rows = {
        screen.M1_MODEL_ID: _metrics(0.30),
        screen.M2_MODEL_ID: _metrics(0.30),
        screen.M4_MODEL_ID: _metrics(0.32),
        screen.M5_MODEL_ID: _metrics(m5_econ, recall=m5_recall),
    }
    return screen.screening_reading(
        rows,
        top10_change_shares={
            "m2_vs_m1": 0.1,
            "m4_vs_m1": 0.5,
            "m5_vs_m4": change_share,
            "m5_vs_m1": 0.6,
        },
        economic_score_ratios={screen.M2_MODEL_ID: 0.2, screen.M5_MODEL_ID: 0.2},
        band_pairs={},
    )


def test_reading_passes_only_when_evaluable_better_than_m4_m1_and_accurate():
    assert _reading(0.34)["classification"] == "directional_pass"
    assert _reading(0.31)["classification"] == "directional_nonpass"
    assert _reading(0.34, change_share=0.0)["classification"] == (
        "not_evaluable_no_top10_change"
    )
    assert _reading(0.34, m5_recall=0.019)["directional_screen_pass"] is False


def test_interaction_is_reported_as_m5_minus_m4_minus_m2_minus_m1():
    reading = _reading(0.34)
    assert np.isclose(
        reading["interaction_m5_minus_m4_minus_m2_minus_m1"][
            "price_purchase_amount_weighted_hit@10"
        ],
        (0.34 - 0.32) - (0.30 - 0.30),
    )
