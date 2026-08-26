import json
import warnings

import numpy as np
import pandas as pd

from lightgcn_clv_information_ceiling_diagnostic import (
    _assign_segments,
    _balanced_accuracy_index,
    _binary_history_features,
    _paired_bootstrap_balanced_index,
    _rank_segment_popularity,
    configure_clv_information_ceiling_diagnostic,
    run_clv_information_ceiling_diagnostic_from_frames,
)
from lightgcn_clv_m3_next_new_transition_diagnostic import (
    _prepare_transactions_frame,
)
from clv_m3_next_new_transition import build_historical_clv


def _transactions() -> pd.DataFrame:
    rows = []
    for user in range(10):
        raw_user = 100 + user
        # Two construction baskets. Item 0 is global; second item varies.
        rows.extend(
            [
                (raw_user, 0, 660, f"{user}-a", 2.0 + user, 2.0 + user),
                (raw_user, 1 + user % 4, 662, f"{user}-b", 4.0 + user, 4.0 + user),
            ]
        )
        # A construction-catalog item that is new to this user in pseudo-future.
        rows.append(
            (raw_user, 1 + (user + 1) % 4, 663, f"{user}-c", 5.0, 5.0)
        )
    # Must be removed before features and item universe are constructed.
    rows.append((100, 999, 670, "future", 999.0, 999.0))
    return pd.DataFrame(
        rows, columns=["u_raw", "i_raw", "t", "b_raw", "v", "up"]
    )


def _prepared(tmp_path):
    cfg = configure_clv_information_ceiling_diagnostic(out_dir=str(tmp_path))
    return _prepare_transactions_frame(_transactions(), cfg.base_config()), cfg


def test_binary_history_features_use_construction_only(tmp_path):
    prepared, _ = _prepared(tmp_path)
    clv, _ = build_historical_clv(
        prepared["construction"], n_users=prepared["n_users"]
    )
    features = _binary_history_features(prepared, clv)

    assert len(features) == 10
    assert set(features.columns) >= {
        "user_idx",
        "unique_items",
        "mean_item_buyer_count",
        "mean_item_price_percentile",
        "clv_percentile",
    }
    assert features["unique_items"].eq(2).all()
    assert 999 not in set(prepared["item_ids"])


def test_segment_assignment_controls_activity_before_clv(tmp_path):
    prepared, _ = _prepared(tmp_path)
    clv, _ = build_historical_clv(
        prepared["construction"], n_users=prepared["n_users"]
    )
    segments = _assign_segments(clv)

    assert set(segments) == {
        "global",
        "n_decile",
        "clv_decile",
        "n_matched_clv",
    }
    assert np.unique(segments["global"]).tolist() == [0]
    # All users have the same N, so the conditional CLV segment must vary only
    # within the same activity segment.
    assert np.unique(segments["n_decile"]).size == 1
    assert np.unique(segments["n_matched_clv"]).size > 1


def test_segment_popularity_excludes_seen_items_and_has_no_backfill():
    user_item = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "i_idx": [0, 1, 0, 2],
        }
    )
    rankings = _rank_segment_popularity(
        user_item,
        segment_by_user=np.array([0, 0]),
        eval_users=np.array([0, 1]),
        seen_items={0: np.array([0]), 1: np.array([0, 2])},
        n_items=4,
        top_k=3,
    )
    np.testing.assert_array_equal(rankings[0], [1, 2])
    np.testing.assert_array_equal(rankings[1], [1])


def test_balanced_accuracy_index_is_geometric_mean_of_six_ratios():
    reference = {f"{metric}@{k}": 1.0 for metric in ("recall", "ndcg") for k in (10, 20, 50)}
    model = dict(reference)
    model["recall@10"] = 2.0
    assert np.isclose(_balanced_accuracy_index(model, reference), 2 ** (1 / 6))
    model["recall@10"] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        assert _balanced_accuracy_index(model, reference) == 0.0
    assert caught == []


def test_paired_bootstrap_balanced_index_is_deterministic_and_paired():
    columns = list({f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)})
    reference = pd.DataFrame({"user_idx": np.arange(20), **{column: np.ones(20) for column in columns}})
    model = reference.copy()
    for column in columns:
        model[column] = 1.1
    interval = _paired_bootstrap_balanced_index(
        model,
        reference,
        comparison_id="model_vs_reference",
        seed=7,
        n_bootstrap=200,
    )
    assert np.isclose(interval["point_estimate"], 1.1)
    assert np.isclose(interval["lo"], 1.1)
    assert np.isclose(interval["hi"], 1.1)
    assert interval["positive_bootstrap_share"] == 1.0


def test_end_to_end_information_diagnostic_writes_traceable_outputs(tmp_path):
    cfg = configure_clv_information_ceiling_diagnostic(out_dir=str(tmp_path))
    result = run_clv_information_ceiling_diagnostic_from_frames(
        _transactions(),
        cfg,
        input_manifest={"synthetic": True},
        source_revision="test-revision",
    )

    assert result.attrs["quality_passed"] is True
    assert set(result.model_id) == {
        "global_popularity",
        "n_decile_popularity",
        "clv_decile_popularity",
        "n_matched_clv_popularity",
    }
    payload = json.loads(open(result.attrs["result_paths"]["json"], encoding="utf-8").read())
    assert payload["source_revision"] == "test-revision"
    assert payload["split"]["transactions_after_day_669"] == 0
    assert payload["metric_contract"]["truth"] == "pseudo-future items absent from each user's construction history"
