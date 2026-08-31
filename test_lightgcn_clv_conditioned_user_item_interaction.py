import numpy as np
import pytest

from lightgcn_clv_conditioned_user_item_interaction import (
    MATCHED_MODEL_ID,
    MODEL_ID,
    build_clv_conditions,
    configure_conditioned_interaction_run,
    preflight_summary,
    screening_reading,
    topk_overlap_summary,
)


def test_preflight_freezes_m2_boundaries_and_matched_control():
    cfg = configure_conditioned_interaction_run(
        out_dir="/tmp/m2-screen",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [MATCHED_MODEL_ID, MODEL_ID]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["m2"]["external_reranking"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.1},
        {"context_dim": 8},
        {"time_cutoff": 697},
        {"diagnostic_max_k": 20},
    ],
)
def test_exploratory_protocol_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        configure_conditioned_interaction_run(
            out_dir="/tmp/m2-screen",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_clv_conditions_separate_overall_level_and_nv_composition():
    axes = {
        "valid_user": np.array([True, True, True, False]),
        "activity_valid": np.array([True, True, True, False]),
        "value_valid": np.array([True, True, True, False]),
        "clv_proxy": np.array([1.0, 4.0, 9.0, 0.0]),
        "q_n": np.array([0.2, 0.8, 0.9, 0.0], np.float32),
        "q_v": np.array([0.9, 0.8, 0.1, 0.0], np.float32),
    }
    q_c, d_nv, valid = build_clv_conditions(axes)

    assert np.all(np.diff(q_c[:3]) > 0)
    np.testing.assert_allclose(d_nv[:3], [-0.7, 0.0, 0.8])
    assert q_c[3] == 0.0 and d_nv[3] == 0.0 and not valid[3]


def test_top10_overlap_is_reported_for_each_clv_segment():
    reference = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32
    )
    model = np.array(
        [[0, 1, 2], [3, 5, 4], [6, 7, 12], [9, 10, 13]], dtype=np.int32
    )
    segments = np.array(["저CLV", "중CLV", "고CLV", "고CLV"])

    table = topk_overlap_summary(reference, model, segments, k=3).set_index("group")

    assert set(table.index) == {"전체", "저CLV", "중CLV", "고CLV"}
    assert table.at["저CLV", "top10_set_changed_user_share"] == 0.0
    assert table.at["중CLV", "top10_set_changed_user_share"] == 0.0
    assert table.at["중CLV", "top10_order_changed_user_share"] == 1.0
    assert table.at["고CLV", "top10_set_changed_user_share"] == 1.0


def _metrics(*, multiplier=1.0, high_delta=0.0, economic_delta=0.0):
    result = {
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
    return result


def test_screen_requires_accuracy_high_clv_economic_and_mechanism_conditions():
    matched = _metrics()
    model = _metrics(multiplier=1.001, high_delta=0.0001, economic_delta=0.0001)
    overlap = topk_overlap_summary(
        np.array([[0, 1], [2, 3], [4, 5]]),
        np.array([[0, 1], [2, 4], [4, 6]]),
        np.array(["저CLV", "중CLV", "고CLV"]),
        k=2,
    )
    reading = screening_reading(
        matched, model, overlap, {"rho_zero_auxiliary_max_abs": 0.0}
    )

    assert reading["positive_screen"] is True
    assert reading["rho0_exact_nonintervention"] is True

    failed = screening_reading(
        matched,
        _metrics(multiplier=0.98, high_delta=0.0001, economic_delta=0.0001),
        overlap,
        {"rho_zero_auxiliary_max_abs": 0.0},
    )
    assert failed["positive_screen"] is False
