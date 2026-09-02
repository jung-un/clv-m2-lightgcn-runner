import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_high_clv_item_response as runner


def test_preflight_freezes_high_clv_item_response_and_protected_split():
    cfg = runner.configure_high_clv_item_response_run(
        out_dir="/tmp/m2-high-clv-response",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        runner.MATCHED_MODEL_ID,
        runner.MODEL_ID,
        runner.SHUFFLED_MODEL_ID,
    ]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["architecture"] == "ID(64)|high-CLV-routed item response(8)"
    assert summary["m2"]["total_dim"] == 72
    assert summary["m2"]["auxiliary_user_layer0"].startswith("exact zero")
    assert summary["m2"]["clv_use"] == "fixed high-segment hard routing"
    assert summary["m2"]["repeatshare_input"] is False
    assert summary["m2"]["item_popularity_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.10},
        {"id_dim": 60},
        {"response_dim": 4},
        {"n_layers": 1},
        {"time_cutoff": 697},
        {"include_degree_matched_shuffle": False},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        runner.configure_high_clv_item_response_run(
            out_dir="/tmp/m2-high-clv-response",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_high_gate_shuffle_preserves_size_but_changes_assignment():
    proxy = np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
    source = np.array([1, 0, 3, 2, 5, 4])

    observed, shuffled, changed = runner._derive_high_gates(proxy, 3.0, source)

    assert observed.sum() == shuffled.sum() == 2
    assert changed > 0
    assert not np.array_equal(observed, shuffled)


def test_high_gate_shuffle_rejects_non_permutation_and_unchanged_gate():
    proxy = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="순열"):
        runner._derive_high_gates(proxy, 2.5, np.array([0, 0, 2, 3]))
    with pytest.raises(RuntimeError, match="배정을 바꾸지"):
        runner._derive_high_gates(proxy, 2.5, np.arange(4))


def _metrics(value=1.0):
    return {
        "recall@10": value,
        "ndcg@10": value,
        "recall@20": value,
        "ndcg@20": value,
        "recall@50": value,
        "ndcg@50": value,
        "고CLV_recall@10": value,
        "고CLV_ndcg@10": value,
        "price_purchase_amount_weighted_hit@10": value,
    }


def test_screening_requires_direct_and_shuffled_gate_attribution():
    matched = _metrics(1.0)
    active = _metrics(1.01)
    shuffled = _metrics(1.005)
    id_only = _metrics(1.004)
    overlap = pd.DataFrame(
        {
            "group": ["전체", "저CLV", "중CLV", "고CLV"],
            "top10_set_changed_user_share": [0.1, 0.0, 0.0, 0.2],
        }
    )
    reading = runner._screening_reading(
        matched,
        active,
        shuffled,
        id_only,
        overlap,
        {"rho_zero_auxiliary_max_abs": 0.0},
    )
    assert reading["positive_screen"] is True

    shuffled["고CLV_ndcg@10"] = 1.02
    failed = runner._screening_reading(
        matched,
        active,
        shuffled,
        id_only,
        overlap,
        {"rho_zero_auxiliary_max_abs": 0.0},
    )
    assert failed["positive_screen"] is False
