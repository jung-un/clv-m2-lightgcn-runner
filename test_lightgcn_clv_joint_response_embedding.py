import numpy as np
import pandas as pd
import pytest

from lightgcn_clv_joint_response_embedding import (
    MATCHED_MODEL_ID,
    MODEL_ID,
    build_item_economic_inputs,
    configure_joint_response_run,
    preflight_summary,
    screening_reading,
)


def test_preflight_freezes_joint_response_m2_and_protected_splits():
    cfg = configure_joint_response_run(
        out_dir="/tmp/m2-joint-response",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [MATCHED_MODEL_ID, MODEL_ID]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["architecture"] == "ID(64)|one joint CLV-response block(4)"
    assert summary["m2"]["total_dim"] == 68
    assert summary["m2"]["repeatshare_input"] is False
    assert summary["m2"]["item_popularity_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.1},
        {"clv_dim": 8},
        {"id_dim": 60},
        {"time_cutoff": 697},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        configure_joint_response_run(
            out_dir="/tmp/m2-joint-response",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_item_economic_inputs_use_overall_and_within_category_price_only():
    train = pd.DataFrame(
        {
            "i_idx": [0, 0, 1, 2, 3],
            "cat_idx": [0, 0, 0, 1, 1],
            "up": [10.0, 20.0, 30.0, 40.0, 80.0],
        }
    )
    features, valid = build_item_economic_inputs(train, n_items=5)

    assert features.shape == (5, 2)
    assert valid.tolist() == [True, True, True, True, False]
    assert features[0, 0] < features[1, 0] < features[2, 0] < features[3, 0]
    assert features[0, 1] < features[1, 1]
    assert features[2, 1] < features[3, 1]
    np.testing.assert_allclose(features[4], [0.0, 0.0])


def _metrics(*, multiplier=1.0, high_delta=0.0, economic_delta=0.0):
    return {
        "recall@10": 0.010 * multiplier,
        "ndcg@10": 0.011 * multiplier,
        "recall@20": 0.020 * multiplier,
        "ndcg@20": 0.021 * multiplier,
        "recall@50": 0.040 * multiplier,
        "ndcg@50": 0.030 * multiplier,
        "고CLV_recall@10": 0.006 + high_delta,
        "고CLV_ndcg@10": 0.007 + high_delta,
        "price_purchase_amount_weighted_hit@10": 0.38 + economic_delta,
    }


def test_screen_requires_accuracy_high_clv_economic_and_actual_change():
    matched = _metrics()
    model = _metrics(multiplier=1.001, high_delta=0.0001, economic_delta=0.0001)
    overlap = pd.DataFrame(
        {
            "group": ["전체", "저CLV", "중CLV", "고CLV"],
            "top10_set_changed_user_share": [0.1, 0.05, 0.1, 0.2],
        }
    )
    reading = screening_reading(
        matched,
        model,
        overlap,
        {"rho_zero_auxiliary_max_abs": 0.0},
    )

    assert reading["positive_screen"] is True

    failed = screening_reading(
        matched,
        _metrics(multiplier=0.98, high_delta=0.0001, economic_delta=0.0001),
        overlap,
        {"rho_zero_auxiliary_max_abs": 0.0},
    )
    assert failed["positive_screen"] is False
