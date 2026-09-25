import numpy as np
import pytest

import clv_m4_m5_top10_movement_diagnostic as diagnostic


def test_truth_movements_and_weighted_delta_reconcile():
    m4 = np.array([np.arange(1, 51), np.arange(100, 150)])
    m5 = np.array([
        np.r_[11, np.arange(2, 11), 1, np.arange(12, 51)],
        np.r_[150, np.arange(101, 150)],
    ])
    items, users = diagnostic.movement_tables(
        [7, 8], ["저CLV", "고CLV"], m4, m5,
        {7: [1, 11, 45], 8: [150]},
        {7: [2.0, 5.0, 7.0], 8: [4.0]},
    )
    by_item = items.set_index(["user", "item"])
    assert by_item.loc[(7, 11), "top10_change"] == "gained"
    assert by_item.loc[(7, 11), "m4_bucket"] == "11-20"
    assert by_item.loc[(7, 1), "top10_change"] == "lost"
    assert by_item.loc[(7, 1), "m5_bucket"] == "11-20"
    assert by_item.loc[(8, 150), "m4_bucket"] == ">50"
    assert users.top10_weight_net.tolist() == [3.0, 4.0]
    summary = diagnostic._summary(items, users).set_index("segment")
    assert summary.loc["전체", "weighted_net"] == 3.5
    assert summary.loc["저CLV", "gained_from_11-20_count"] == 1
    assert summary.loc["고CLV", "gained_from_>50_count"] == 1
    assert summary.loc["전체", "recall_net"] == pytest.approx((0 + 1) / 2)


def test_duplicate_candidates_or_truth_are_rejected():
    top = np.arange(1, 51)[None, :]
    duplicate = top.copy()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(ValueError, match="Duplicate candidate"):
        diagnostic.movement_tables([7], ["전체"], top, duplicate,
                                   {7: [1]}, {7: [2.0]})
    with pytest.raises(ValueError, match="Duplicate held-out truth"):
        diagnostic.movement_tables([7], ["전체"], top, top,
                                   {7: [1, 1]}, {7: [2.0, 3.0]})


def test_unapproved_result_file_is_rejected_before_loading(tmp_path):
    fake = tmp_path / "result.json"
    fake.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing/changed"):
        diagnostic.run(fake, tmp_path / "out")
