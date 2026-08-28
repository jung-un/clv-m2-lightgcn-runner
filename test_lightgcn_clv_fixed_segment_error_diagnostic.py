import numpy as np
import pandas as pd
from types import SimpleNamespace

import lightgcn_clv_fixed_segment_error_diagnostic as diagnostic


def test_item_role_occurrences_separates_hits_misses_and_false_positives():
    frame = diagnostic.item_role_occurrences(
        users=np.array([0]),
        segments=np.array([diagnostic.SEGMENT_ORDER[1]]),
        truth={0: np.array([2, 11, 99])},
        top50=np.array([[2, *range(3, 11), 12, 11, *range(20, 58)]]),
        truth_amount={0: np.array([5.0, 7.0, 9.0])},
    )

    roles = {
        role: set(group.item_idx.tolist())
        for role, group in frame.groupby("role", sort=False)
    }
    assert roles["truth_hit_top10"] == {2}
    assert roles["truth_miss_top10"] == {11, 99}
    assert roles["truth_rank_11_20"] == {11}
    assert roles["truth_rank_over_50"] == {99}
    assert roles["false_positive_top10"] == set(range(3, 11)) | {12}


def test_attach_history_relations_marks_category_overlap_and_embedding_similarity():
    occurrences = pd.DataFrame(
        {
            "user_idx": [0, 0],
            "item_idx": [1, 2],
            "category": ["bread", "drink"],
        }
    )
    train = pd.DataFrame(
        {
            "u_idx": [0, 0],
            "i_idx": [0, 3],
            "cat_raw": ["bread", "snack"],
        }
    )
    item_embedding = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )

    result = diagnostic.attach_history_relations(
        occurrences,
        train=train,
        item_embedding=item_embedding,
        n_users=1,
    )

    assert result.history_category_overlap.tolist() == [1.0, 0.0]
    assert result.history_embedding_cosine.tolist() == [1.0, 0.0]


def test_segment_role_summary_and_contrast_compare_misses_to_false_positives():
    frame = pd.DataFrame(
        {
            "segment": [
                diagnostic.SEGMENT_ORDER[0],
                diagnostic.SEGMENT_ORDER[0],
                diagnostic.SEGMENT_ORDER[2],
                diagnostic.SEGMENT_ORDER[2],
            ],
            "role": [
                "truth_miss_top10",
                "false_positive_top10",
                "truth_miss_top10",
                "false_positive_top10",
            ],
            "item_idx": [1, 2, 3, 4],
            "price_percentile": [0.2, 0.5, 0.9, 0.4],
            "train_user_count": [10.0, 100.0, 20.0, 80.0],
            "repeat_purchase_share": [0.1, 0.6, 0.2, 0.5],
            "history_category_overlap": [1.0, 0.0, 0.0, 1.0],
            "history_embedding_cosine": [0.8, 0.2, 0.3, 0.7],
        }
    )

    summary = diagnostic.summarize_segment_item_roles(frame)
    contrast = diagnostic.miss_false_positive_contrasts(summary)

    lookup = contrast.set_index(["segment", "trait"])
    assert lookup.at[(diagnostic.SEGMENT_ORDER[0], "price_percentile"), "miss_minus_false_positive"] == -0.3
    assert lookup.at[(diagnostic.SEGMENT_ORDER[2], "price_percentile"), "miss_minus_false_positive"] == 0.5
    assert lookup.at[(diagnostic.SEGMENT_ORDER[0], "train_user_count"), "miss_minus_false_positive"] == -90.0


def test_preflight_uses_only_historical_development_checkpoint(tmp_path):
    cfg = diagnostic.configure_fixed_segment_error_diagnostic(
        out_dir=str(tmp_path / "diagnostic"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    summary = diagnostic.preflight_summary(cfg)

    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["split"] == "historical_development_days_684_690"
    assert summary["fixed_clv_source"] == "train-history N×V proxy at day 683"


def test_segments_for_users_maps_global_user_ids_through_eval_user_order():
    cache = SimpleNamespace(
        users=np.array([100, 1764, 2499]),
        seg=np.array(["low", "mid", "high"]),
    )

    result = diagnostic.segments_for_users(
        cache,
        np.array([1764, 100]),
    )

    assert result.tolist() == ["mid", "low"]


def test_segment_order_reuses_the_canonical_evaluation_labels():
    assert diagnostic.SEGMENT_ORDER == tuple(diagnostic.v3.SEG_NAMES)
