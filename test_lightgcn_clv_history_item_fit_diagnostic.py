import numpy as np
import pandas as pd
import pytest
import torch

import lightgcn_clv_history_item_fit_diagnostic as diagnostic


def test_raw_item_traits_exposes_item_idx_for_downstream_merges():
    train = pd.DataFrame(
        {
            "u_idx": [0, 1],
            "i_idx": [0, 1],
            "i_raw": ["item-a", "item-b"],
            "cat_raw": ["cat-a", "cat-b"],
            "b_raw": ["basket-a", "basket-b"],
            "up": [1.0, 2.0],
        }
    )

    traits = diagnostic._raw_item_traits(train, n_items=2)

    assert traits["item_idx"].tolist() == [0, 1]
    assert traits["item_id"].tolist() == ["item-a", "item-b"]


def test_axis_views_slice_exact_trained_blocks():
    user = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
    item = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)

    views = diagnostic.axis_views(user, item, id_dim=4, axis_dim=2)

    assert tuple(views) == ("id_only", "id_n", "id_v", "full")
    torch.testing.assert_close(views["id_only"][0], user[:, :4])
    torch.testing.assert_close(views["id_v"][0], torch.cat([user[:, :4], user[:, 6:]], 1))
    torch.testing.assert_close(views["full"][1], item)


def test_rank_transition_table_tracks_each_truth_item_by_segment():
    table = diagnostic.rank_transition_table(
        users=np.array([0, 1]),
        segments=np.array(["저CLV", "고CLV"]),
        truth={0: np.array([10, 11]), 1: np.array([20])},
        reference_top50=np.array(
            [[10, *range(100, 113), 11, *range(200, 235)], range(300, 350)],
            dtype=np.int64,
        ),
        model_top50=np.array(
            [[11, *range(100, 110), 10, *range(200, 238)], [*range(300, 339), 20, *range(400, 410)]],
            dtype=np.int64,
        ),
    )

    lookup = table.set_index(["segment", "reference_bucket", "model_bucket"])
    assert lookup.at[("저CLV", "1-10", "11-20"), "truth_item_count"] == 1
    assert lookup.at[("저CLV", "11-20", "1-10"), "truth_item_count"] == 1
    assert lookup.at[("고CLV", ">50", "21-50"), "truth_item_count"] == 1
    assert table.groupby("segment")["share_within_segment"].sum().to_dict() == {
        "고CLV": pytest.approx(1.0),
        "저CLV": pytest.approx(1.0),
    }


def test_item_role_occurrences_identify_promoted_and_displaced_items():
    frame = diagnostic.item_role_occurrences(
        users=np.array([0]),
        segments=np.array(["중CLV"]),
        truth={0: np.array([2, 99])},
        m1_top50=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, *range(20, 60)]]),
        m2_top50=np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, *range(20, 60)]]),
    )

    promoted = frame[frame.role.eq("m2_promoted_top10")]
    displaced = frame[frame.role.eq("m1_displaced_top10")]
    assert promoted.item_idx.tolist() == [11]
    assert displaced.item_idx.tolist() == [1]
    assert bool(promoted.is_truth_item.iloc[0]) is False


def test_preflight_is_descriptive_and_never_trains(tmp_path):
    cfg = diagnostic.configure_history_item_fit_diagnostic(
        out_dir=str(tmp_path / "m2"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    summary = diagnostic.preflight_summary(cfg)

    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["views"] == ["id_only", "id_n", "id_v", "full"]
    assert summary["statistical_note"] == "descriptive checkpoint diagnostic; no significance claim"
