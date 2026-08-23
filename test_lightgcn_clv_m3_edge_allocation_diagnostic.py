import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m3_edge_allocation_diagnostic as diagnostic


def test_intervention_table_measures_edge_ratio_and_item_dispersion():
    edges = diagnostic._edge_intervention_table(
        edge_users=np.array([0, 1, 2, 0]),
        edge_items=np.array([0, 0, 0, 1]),
        base=np.array([0.2, 0.2, 0.2, 0.5]),
        adjusted=np.array([0.1, 0.2, 0.3, 0.5]),
        relationship_share=np.array([0.75, 1.0, 1.0, 0.25]),
        allocation=np.array([6.0, 4.0, 2.0, 2.0]),
    )
    assert np.allclose(edges.coefficient_ratio, [0.5, 1.0, 1.5, 1.0])
    assert np.allclose(edges.absolute_coefficient_change, [0.1, 0.0, 0.1, 0.0])
    item_zero = edges[edges.item_idx.eq(0)]
    assert np.isclose(item_zero.adjusted_coefficient.sum(), 0.6)

    summary = diagnostic._intervention_summary(edges)
    item_row = summary[
        summary.grain.eq("item") & summary.group_id.eq("all_items")
    ].iloc[0]
    assert item_row.n_entities == 2
    assert np.isclose(item_row.share_edges_changed, 0.5)
    assert item_row.median_within_entity_ratio_std > 0


def test_candidate_connectivity_uses_only_training_relations():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 1, 2, 2],
            "i_idx": [0, 1, 2, 0, 3, 4, 1, 3],
            "b_raw": ["a", "a", "b", "c", "c", "d", "e", "e"],
            "t": [1, 1, 2, 1, 1, 2, 1, 1],
        }
    )
    candidates = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "item_idx": [3, 4, 5],
            "role": ["truth", "recommendation", "truth"],
        }
    )
    item_categories = np.array(["A", "A", "B", "A", "C", "B"], dtype=object)

    result = diagnostic._candidate_connectivity(
        train=train,
        candidates=candidates,
        item_categories=item_categories,
        n_users=3,
        n_items=6,
    ).set_index("item_idx")

    # Item 3 shares buyers with user 0's history, co-occurs with item 1,
    # and belongs to a category already present in that history.
    assert result.loc[3, "shared_buyer_reach"] > 0
    assert result.loc[3, "co_basket_reach"] > 0
    assert result.loc[3, "history_category_share"] == 2 / 3
    # Item 4 follows a basket containing item 0 for another training user.
    assert result.loc[4, "forward_transition_reach"] > 0
    # An unseen item has no graph-based route, even if its category is known.
    assert result.loc[5, "shared_buyer_reach"] == 0
    assert result.loc[5, "co_basket_reach"] == 0
    assert result.loc[5, "forward_transition_reach"] == 0


def test_structure_evidence_compares_truth_with_added_recommendations():
    connectivity = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1],
            "item_idx": [3, 4, 5, 6, 7, 8],
            "role": ["truth", "m1_only", "m3_only"] * 2,
            "shared_buyer_reach": [0.8, 0.2, 0.3, 0.7, 0.1, 0.2],
            "co_basket_reach": [0.9, 0.1, 0.2, 0.8, 0.2, 0.3],
            "forward_transition_reach": [0.6, 0.0, 0.1, 0.5, 0.1, 0.0],
            "history_category_share": [0.7, 0.2, 0.4, 0.6, 0.1, 0.3],
        }
    )
    evidence = diagnostic._structure_evidence(connectivity)
    co_basket = evidence[evidence.signal.eq("co_basket_reach")].iloc[0]
    assert np.isclose(co_basket.truth_mean, 0.85)
    assert np.isclose(co_basket.m3_only_mean, 0.25)
    assert np.isclose(co_basket.truth_minus_m3_only, 0.60)
    assert co_basket.direction == "truth_stronger"


def test_rank_movement_summary_keeps_top10_and_top50_separate():
    truth = pd.DataFrame(
        {
            "m1_rank_capped_101": [9, 11, 51, 101],
            "m3_rank_capped_101": [12, 8, 40, 80],
        }
    )
    summary = diagnostic._rank_movement_summary(truth).set_index("cutoff")
    assert summary.loc[10, "entered"] == 1
    assert summary.loc[10, "left"] == 1
    assert summary.loc[50, "entered"] == 1
    assert summary.loc[50, "left"] == 0
    assert summary.loc[100, "entered"] == 1


def test_candidate_rows_label_truth_and_model_only_items_by_cutoff():
    truth = pd.DataFrame(
        {"user_idx": [0, 0], "item_idx": [8, 9], "test_purchase_amount": [3.0, 4.0]}
    )
    recommendations = pd.DataFrame(
        {
            "user_idx": [0] * 8,
            "model_id": [diagnostic.M1_ID] * 4 + [diagnostic.M3_ID] * 4,
            "rank": [1, 2, 11, 12, 1, 2, 11, 12],
            "item_idx": [1, 2, 3, 4, 2, 5, 3, 6],
        }
    )
    candidates = diagnostic._candidate_rows(truth, recommendations)
    top10 = candidates[candidates.cutoff.eq(10)]
    assert set(top10[top10.role.eq("truth")].item_idx) == {8, 9}
    assert set(top10[top10.role.eq("m1_only")].item_idx) == {1}
    assert set(top10[top10.role.eq("m3_only")].item_idx) == {5}
    top50 = candidates[candidates.cutoff.eq(50)]
    assert set(top50[top50.role.eq("m1_only")].item_idx) == {1, 4}
    assert set(top50[top50.role.eq("m3_only")].item_idx) == {5, 6}


def test_preflight_forbids_training_and_final_test(tmp_path):
    out_dir = tmp_path / "results_m3_clv_edge_allocation_historical_dunnhumby"
    cfg = diagnostic.configure_m3_edge_allocation_diagnostic(out_dir=str(out_dir))
    summary = diagnostic.preflight_summary(cfg)
    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["final_test_constructed"] is False
    assert summary["source_split"] == "DAY 1--683 train; DAY 684--690 evaluation"
    with pytest.raises(ValueError, match="historical"):
        diagnostic.configure_m3_edge_allocation_diagnostic(
            out_dir=str(tmp_path / "results_m3_final_test")
        )
