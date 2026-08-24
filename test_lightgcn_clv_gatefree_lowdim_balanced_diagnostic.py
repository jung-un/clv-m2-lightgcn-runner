import numpy as np
import pytest
import pandas as pd

import lightgcn_clv_gatefree_lowdim_balanced_diagnostic as diagnostic


def test_value_weighted_transition_preserves_count_and_exact_weight():
    table = diagnostic.value_weighted_rank_transition_table(
        users=np.array([0, 1]),
        segments=np.array(["저CLV", "고CLV"]),
        truth={0: np.array([10, 11]), 1: np.array([20])},
        truth_weights={0: np.array([2.0, 8.0]), 1: np.array([5.0])},
        reference_ranks={0: {10: 3, 11: 15}, 1: {20: 55}},
        model_ranks={0: {10: 12, 11: 7}, 1: {20: 40}},
    )

    lookup = table.set_index(["segment", "reference_bucket", "model_bucket"])
    assert lookup.at[("저CLV", "1-10", "11-20"), "truth_weight_sum"] == 2.0
    assert lookup.at[("저CLV", "11-20", "1-10"), "truth_weight_sum"] == 8.0
    assert lookup.at[("고CLV", ">50", "21-50"), "truth_weight_sum"] == 5.0
    assert table["truth_item_count"].sum() == 3
    assert table["truth_weight_sum"].sum() == 15.0
    assert table.groupby("segment")["truth_weight_share_within_segment"].sum().to_dict() == {
        "고CLV": pytest.approx(1.0),
        "저CLV": pytest.approx(1.0),
    }


def test_weighted_movement_summary_distinguishes_promoted_and_demoted_value():
    transition = pd.DataFrame(
        [
            {
                "segment": "중CLV",
                "bucket_movement": "promoted",
                "truth_item_count": 1,
                "truth_weight_sum": 9.0,
            },
            {
                "segment": "중CLV",
                "bucket_movement": "demoted",
                "truth_item_count": 3,
                "truth_weight_sum": 1.0,
            },
        ]
    )

    summary = diagnostic.weighted_movement_summary(transition).set_index(
        "bucket_movement"
    )

    assert summary.at["promoted", "truth_item_share_within_segment"] == pytest.approx(0.25)
    assert summary.at["promoted", "truth_weight_share_within_segment"] == pytest.approx(0.9)
    assert summary.at["demoted", "truth_weight_share_within_segment"] == pytest.approx(0.1)


def test_axis_attribution_uses_internal_id_only_as_reference():
    metrics = pd.DataFrame(
        [
            {
                "view": "id_only",
                "recall@10": 0.1,
                "ndcg@10": 0.2,
                "price_purchase_amount_weighted_hit@10": 2.0,
            },
            {
                "view": "id_n",
                "recall@10": 0.11,
                "ndcg@10": 0.2,
                "price_purchase_amount_weighted_hit@10": 2.1,
            },
            {
                "view": "id_v",
                "recall@10": 0.1,
                "ndcg@10": 0.22,
                "price_purchase_amount_weighted_hit@10": 2.3,
            },
            {
                "view": "full",
                "recall@10": 0.12,
                "ndcg@10": 0.23,
                "price_purchase_amount_weighted_hit@10": 2.4,
            },
        ]
    )

    table = diagnostic.axis_attribution_table(metrics)
    lookup = table.set_index(["view", "metric"])

    assert lookup.at[("id_n", "recall@10"), "absolute_delta"] == pytest.approx(0.01)
    assert lookup.at[
        ("id_v", "price_purchase_amount_weighted_hit@10"), "relative_change_pct"
    ] == pytest.approx(15.0)
    assert set(table["reference_view"]) == {"id_only"}


def test_preflight_is_checkpoint_only_and_uses_existing_truth_weights(tmp_path):
    cfg = diagnostic.configure_balanced_checkpoint_diagnostic(out_dir=str(tmp_path))
    summary = diagnostic.preflight_summary(cfg)

    assert summary["training"] is False
    assert summary["model_selection"] is False
    assert summary["seed"] == 42
    assert "EvalCache.rev" in summary["truth_weight_source"]
    assert "no significance" in summary["statistical_note"]
