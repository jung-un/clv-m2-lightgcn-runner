import numpy as np
import pytest

from lightgcn_clv_m4_clv_hard_negative import (
    K1_MODEL_ID,
    MEAN_K5_MODEL_ID,
    M4_MODEL_ID,
    configure_m4_clv_hard_negative_run,
    preflight_summary,
    sample_uniform_negative_matrix,
    screening_reading,
)


def test_preflight_freezes_loss_only_development_screen():
    cfg = configure_m4_clv_hard_negative_run(
        out_dir="/tmp/m4-clv-hard",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [MEAN_K5_MODEL_ID, M4_MODEL_ID]
    assert summary["reused_comparator"] == K1_MODEL_ID
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m4"]["uniform_negative_count"] == 5
    assert summary["m4"]["per_positive_loss_mass"] == 1.0
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["m2_representation"] is False
    assert summary["fixed"]["m3_edge_weight"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"negative_count": 4},
        {"time_cutoff": 697},
        {"id_dim": 96},
        {"epochs": 99},
    ],
)
def test_config_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M4 screen"):
        configure_m4_clv_hard_negative_run(
            out_dir="/tmp/m4-clv-hard",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_uniform_negative_matrix_is_reproducible_and_excludes_train_pairs():
    users = np.array([0, 1], dtype=np.int64)
    positives = np.array([0, 2], dtype=np.int64)
    n_items = 5
    positive_keys = np.array([0, 1, 7, 8], dtype=np.int64)

    first = sample_uniform_negative_matrix(
        users,
        positives,
        n_items,
        positive_keys,
        np.random.default_rng(7),
        k=5,
    )
    second = sample_uniform_negative_matrix(
        users,
        positives,
        n_items,
        positive_keys,
        np.random.default_rng(7),
        k=5,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 5)
    sampled_keys = users[:, None] * n_items + first
    assert not np.isin(sampled_keys, positive_keys).any()


def _metrics(*, multiplier=1.0, high_delta=0.0, economic_delta=0.0):
    return {
        "recall@10": 0.01 * multiplier,
        "ndcg@10": 0.011 * multiplier,
        "recall@20": 0.02 * multiplier,
        "ndcg@20": 0.021 * multiplier,
        "recall@50": 0.04 * multiplier,
        "ndcg@50": 0.03 * multiplier,
        "고CLV_recall@10": 0.006 + high_delta,
        "고CLV_ndcg@10": 0.007 + high_delta,
        "price_purchase_amount_weighted_hit@10": 0.38 + economic_delta,
        "coverage@10": 0.0034,
        "n_distinct@10": 306.0,
        "top10_share@10": 0.36,
    }


def test_screen_requires_accuracy_high_clv_economic_and_exposure_guards():
    baseline = _metrics()
    mean_k5 = _metrics(multiplier=1.001)
    m4 = _metrics(multiplier=1.002, high_delta=0.0001, economic_delta=0.0001)

    reading = screening_reading(baseline, mean_k5, m4)

    assert reading["positive_screen"] is True
    assert screening_reading(
        baseline,
        mean_k5,
        _metrics(multiplier=0.98, high_delta=0.0001, economic_delta=0.0001),
    )["positive_screen"] is False
