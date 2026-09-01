import json

import numpy as np
import pytest
import torch

from clv_run_state import file_sha256
import lightgcn_clv_m3_category_transition_diagnostic as diagnostic
from lightgcn_clv_m3_category_transition_diagnostic import (
    clv_assignment_correlation,
    configure_m3_category_transition_diagnostic,
    graph_similarity_summary,
    preflight_summary,
    recommendation_overlap_summary,
    score_component_long_summary,
    score_component_summary,
    score_pairs_from_parts,
    sparse_row_similarity,
)


def test_load_arm_model_accepts_source_checkpoint_seed_inside_config(
    tmp_path, monkeypatch
):
    cfg = configure_m3_category_transition_diagnostic(
        source_out_dir=str(tmp_path / "source"),
        diagnostic_out_dir=str(tmp_path / "diagnostic"),
    )
    model_id = "actual_model"
    arm = "actual_clv"
    record_dir = (
        tmp_path / "source" / "arms" / cfg.source_result_id
    )
    record_dir.mkdir(parents=True)
    checkpoint = record_dir / f"{model_id}_s42.pt"
    source_model = torch.nn.Linear(2, 1)
    torch.save(
        {
            "state": source_model.state_dict(),
            "model_id": model_id,
            "graph_arm": arm,
            "config": {"seed": 42},
            "training": {"final_epoch": 100},
            "source_revision": "source-revision",
            "input_hash": "input-hash",
        },
        checkpoint,
    )
    record = {
        "model_id": model_id,
        "graph_arm": arm,
        "seed": 42,
        "input_hash": "input-hash",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    (record_dir / f"{model_id}_s42.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    monkeypatch.setattr(
        diagnostic.source_runner,
        "_build_model",
        lambda *_args: torch.nn.Linear(2, 1),
    )

    loaded, loaded_record, loaded_path = diagnostic._load_arm_model(
        cfg,
        {"input_hash": "input-hash"},
        object(),
        arm=arm,
        model_id=model_id,
    )

    assert loaded_path == checkpoint
    assert loaded_record == record
    assert torch.equal(loaded.weight, source_model.weight)


def test_preflight_locks_checkpoint_only_historical_diagnostic(tmp_path):
    cfg = configure_m3_category_transition_diagnostic(
        source_out_dir=str(tmp_path / "source"),
        diagnostic_out_dir=str(tmp_path / "diagnostic"),
    )
    summary = preflight_summary(cfg)
    assert summary["code_version"] == (
        "m3-clv-category-transition-failure-diagnostic-v2"
    )
    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["source_split"] == "DAY 1--683 train; DAY 684--690 evaluation"
    assert summary["final_test_constructed"] is False
    assert summary["holdout_constructed"] is False
    assert summary["rank_limit"] == 50
    assert summary["source_models"] == [
        "m3_clv_conditioned_first_acquisition_category_transition",
        "m3_clv_conditioned_first_acquisition_category_transition_shuffle",
    ]


def _sparse(rows, cols, values, shape):
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            torch.tensor([rows, cols], dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
            size=shape,
        ).coalesce()


def test_sparse_row_similarity_detects_assignment_and_weight_changes():
    actual = _sparse(
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [0.75, 0.25, 0.5, 0.5],
        (2, 4),
    )
    shuffled = _sparse(
        [0, 0, 1, 1],
        [0, 2, 1, 2],
        [0.5, 0.5, 0.5, 0.5],
        (2, 4),
    )

    result = sparse_row_similarity(
        actual,
        shuffled,
        q_actual=np.array([0.1, 0.9]),
        q_shuffle=np.array([0.9, 0.1]),
        strata=np.array([0, 0]),
    ).set_index("user_idx")

    assert result.loc[0, "common_edge_count"] == 1
    assert result.loc[0, "edge_jaccard"] == pytest.approx(1 / 3)
    assert result.loc[0, "total_variation_distance"] == pytest.approx(0.5)
    assert result.loc[0, "weight_cosine"] == pytest.approx(0.6708203932)
    assert bool(result.loc[0, "exact_relation_row"]) is False
    assert result.loc[1, "edge_jaccard"] == pytest.approx(1.0)
    assert result.loc[1, "total_variation_distance"] == pytest.approx(0.0)
    assert result.loc[1, "weight_cosine"] == pytest.approx(1.0)
    assert bool(result.loc[1, "exact_relation_row"]) is True

    summary = graph_similarity_summary(result.reset_index()).set_index("clv_group")
    assert summary.loc["전체", "n_users"] == 2
    assert summary.loc["전체", "mean_edge_jaccard"] == pytest.approx(2 / 3)
    assert summary.loc["전체", "median_total_variation_distance"] == pytest.approx(0.25)
    assert summary.loc["전체", "exact_relation_user_share"] == pytest.approx(0.5)

    correlation = clv_assignment_correlation(result.reset_index()).set_index("scope")
    assert correlation.loc["전체", "n_users"] == 2
    assert correlation.loc["전체", "q_actual_shuffle_spearman"] == pytest.approx(-1.0)
    assert correlation.loc["degree_stratum_0", "q_actual_shuffle_spearman"] == pytest.approx(-1.0)


def test_recommendation_overlap_is_reported_overall_and_by_clv_quintile():
    reference = np.array([[1, 2, 3], [4, 5, 6]])
    model = np.array([[1, 3, 7], [4, 5, 6]])

    summary = recommendation_overlap_summary(
        reference,
        model,
        q_actual=np.array([0.1, 0.9]),
        comparison="actual_vs_shuffle",
        ks=(2, 3),
    ).set_index(["comparison", "clv_group", "k"])

    overall_k2 = summary.loc[("actual_vs_shuffle", "전체", 2)]
    assert overall_k2["n_users"] == 2
    assert overall_k2["set_changed_user_share"] == pytest.approx(0.5)
    assert overall_k2["order_changed_user_share"] == pytest.approx(0.5)
    assert overall_k2["mean_jaccard"] == pytest.approx(2 / 3)
    assert summary.loc[("actual_vs_shuffle", "Q1", 2), "mean_jaccard"] == pytest.approx(1 / 3)
    assert summary.loc[("actual_vs_shuffle", "Q5", 3), "mean_jaccard"] == pytest.approx(1.0)


def test_score_component_summary_uses_the_same_candidate_pairs():
    base = np.array([[1.0, 2.0], [3.0, 5.0]])
    auxiliary = np.array([[0.1, -0.1], [0.2, 0.4]])

    summary = score_component_summary(
        base,
        auxiliary,
        q_actual=np.array([0.1, 0.9]),
        model_id="actual",
        candidate_role="common_top50_union",
    ).set_index(["model_id", "candidate_role", "clv_group"])

    overall = summary.loc[("actual", "common_top50_union", "전체")]
    assert overall["candidate_pair_count"] == 4
    assert overall["base_score_std"] == pytest.approx(np.sqrt(2.1875))
    assert overall["auxiliary_score_std"] == pytest.approx(np.sqrt(0.0325))
    assert overall["auxiliary_to_base_std_ratio"] == pytest.approx(
        np.sqrt(0.0325 / 2.1875)
    )
    assert overall["mean_abs_auxiliary_score"] == pytest.approx(0.2)
    assert summary.loc[("actual", "common_top50_union", "Q1"), "candidate_pair_count"] == 2


def test_score_component_long_summary_keeps_variable_truth_counts():
    summary = score_component_long_summary(
        base_scores=np.array([1.0, 2.0, 3.0]),
        auxiliary_scores=np.array([0.1, -0.1, 0.2]),
        score_users=np.array([0, 0, 1]),
        q_by_user=np.array([0.1, 0.9]),
        model_id="actual",
        candidate_role="heldout_truth",
    ).set_index(["model_id", "candidate_role", "clv_group"])

    assert summary.loc[("actual", "heldout_truth", "전체"), "candidate_pair_count"] == 3
    assert summary.loc[("actual", "heldout_truth", "Q1"), "candidate_pair_count"] == 2
    assert summary.loc[("actual", "heldout_truth", "Q5"), "candidate_pair_count"] == 1
    assert summary.loc[("actual", "heldout_truth", "전체"), "auxiliary_score_mean"] == pytest.approx(0.2 / 3)


def test_score_pairs_from_parts_matches_model_score_decomposition():
    base_user = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    base_item = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    message = torch.tensor([[0.5, 1.0], [1.0, -1.0]])

    base, auxiliary = score_pairs_from_parts(
        base_user,
        base_item,
        message,
        users=np.array([0, 1]),
        items=np.array([1, 0]),
        gamma=0.1,
        batch_size=1,
    )

    assert base.tolist() == pytest.approx([4.0, 3.0])
    assert auxiliary.tolist() == pytest.approx([0.2, 0.1])
