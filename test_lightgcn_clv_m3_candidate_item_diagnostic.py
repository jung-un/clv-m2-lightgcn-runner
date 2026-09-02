import numpy as np
import pandas as pd
import torch
import json

from clv_m3_clv_conditioned_candidate_item_graph import (
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
)
import lightgcn_clv_m3_candidate_item_diagnostic as diagnostic
import lightgcn_clv_m3_clv_conditioned_candidate_item as source_runner


def _operator(rows, n_items=8):
    user_indices = []
    item_indices = []
    values = []
    for user, relation in enumerate(rows):
        for item, value in relation.items():
            user_indices.append(user)
            item_indices.append(item)
            values.append(value)
    return torch.sparse_coo_tensor(
        torch.tensor([user_indices, item_indices], dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        size=(len(rows), n_items),
    ).coalesce()


def test_preflight_is_checkpoint_only_and_routes_three_stages(tmp_path):
    cfg = diagnostic.configure_m3_candidate_item_diagnostic(
        source_out_dir=str(tmp_path / "source"),
        diagnostic_out_dir=str(tmp_path / "diagnostic"),
    )
    summary = diagnostic.preflight_summary(cfg)
    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["final_test_constructed"] is False
    assert summary["holdout_constructed"] is False
    assert summary["source_result_id"] == "d5c0423bfd90"
    assert set(summary["routing_rule"]) == {
        "candidate_relation_construction",
        "relation_to_score_transfer",
        "score_to_rank_boundary",
        "ranking_alignment",
    }


def test_source_result_requires_development_split_without_holdout(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cfg = diagnostic.configure_m3_candidate_item_diagnostic(
        source_result_id="abc123",
        source_out_dir=str(source_dir),
        diagnostic_out_dir=str(tmp_path / "diagnostic"),
    )
    payload = {
        "code_version": source_runner.CODE_VERSION,
        "config": {"seed": 42},
        "preflight": {
            "historical_development_split": {
                "final_test_constructed": False,
                "holdout_constructed": False,
            }
        },
    }
    path = source_dir / "m3_clv_candidate_item_abc123.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded_path, loaded = diagnostic._source_result(cfg)
    assert loaded_path == path
    assert loaded == payload


def test_candidate_truth_coverage_separates_micro_macro_and_weight():
    operators = {
        ARM_GENERAL: _operator(
            [{1: 0.6, 2: 0.4}, {3: 0.5, 4: 0.5}]
        ),
        ARM_ACTUAL: _operator(
            [{1: 0.7, 5: 0.3}, {3: 0.6, 6: 0.4}]
        ),
        ARM_SHUFFLE: _operator(
            [{2: 0.8, 5: 0.2}, {4: 0.7, 6: 0.3}]
        ),
    }
    per_user, summary = diagnostic.candidate_truth_coverage(
        operators,
        evaluation_users=np.array([0, 1]),
        truths={0: {1, 7}, 1: {3}},
        q_actual=np.array([0.1, 0.9]),
    )
    actual = summary.loc[
        summary["graph_arm"].eq(ARM_ACTUAL)
        & summary["clv_group"].eq("전체")
    ].iloc[0]
    shuffled = summary.loc[
        summary["graph_arm"].eq(ARM_SHUFFLE)
        & summary["clv_group"].eq("전체")
    ].iloc[0]
    assert len(per_user) == 6
    assert actual["candidate_truth_hits"] == 2
    assert np.isclose(actual["candidate_truth_pair_coverage"], 2 / 3)
    assert np.isclose(actual["macro_candidate_truth_recall"], 0.75)
    assert np.isclose(actual["mean_truth_edge_weight_all_truth"], 1.3 / 3)
    assert shuffled["candidate_truth_hits"] == 0


def _route_inputs(
    *,
    actual_coverage,
    shuffle_coverage,
    actual_score,
    shuffle_score,
    set_changed,
):
    candidates = pd.DataFrame(
        [
            {
                "graph_arm": ARM_GENERAL,
                "clv_group": "전체",
                "candidate_truth_pair_coverage": 0.1,
                "macro_candidate_truth_recall": 0.1,
                "mean_truth_edge_weight_all_truth": 0.001,
            },
            {
                "graph_arm": ARM_ACTUAL,
                "clv_group": "전체",
                "candidate_truth_pair_coverage": actual_coverage,
                "macro_candidate_truth_recall": actual_coverage,
                "mean_truth_edge_weight_all_truth": actual_coverage / 100,
            },
            {
                "graph_arm": ARM_SHUFFLE,
                "clv_group": "전체",
                "candidate_truth_pair_coverage": shuffle_coverage,
                "macro_candidate_truth_recall": shuffle_coverage,
                "mean_truth_edge_weight_all_truth": shuffle_coverage / 100,
            },
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "model_id": source_runner.ACTUAL_ID,
                "clv_group": "전체",
                "truth_minus_top50_auxiliary_mean": actual_score,
            },
            {
                "model_id": source_runner.SHUFFLE_ID,
                "clv_group": "전체",
                "truth_minus_top50_auxiliary_mean": shuffle_score,
            },
        ]
    )
    overlaps = pd.DataFrame(
        [
            {
                "comparison": "actual_full_vs_shuffle_full",
                "clv_group": "전체",
                "k": k,
                "set_changed_user_share": set_changed,
            }
            for k in (10, 20, 50)
        ]
    )
    return candidates, scores, overlaps


def test_routing_changes_only_the_failed_stage():
    cases = [
        ((0.10, 0.11, 0.2, 0.1, 0.1), "candidate_relation_construction"),
        ((0.12, 0.11, 0.1, 0.2, 0.1), "relation_to_score_transfer"),
        ((0.12, 0.11, 0.2, 0.1, 0.0), "score_to_rank_boundary"),
        ((0.12, 0.11, 0.2, 0.1, 0.1), "ranking_alignment"),
    ]
    for arguments, expected in cases:
        reading = diagnostic.diagnostic_route(
            *_route_inputs(
                actual_coverage=arguments[0],
                shuffle_coverage=arguments[1],
                actual_score=arguments[2],
                shuffle_score=arguments[3],
                set_changed=arguments[4],
            ),
            source_attribution_supported=False,
        )
        assert reading["descriptive_bottleneck"] == expected
        assert reading["automatic_model_selection"] is False
