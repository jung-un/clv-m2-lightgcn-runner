import numpy as np
import pytest
import torch

import lightgcn_clv_gatefree_lowdim_diagnostic as diagnostic


def test_axis_views_select_only_requested_propagated_blocks():
    user = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
    item = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)

    views = diagnostic.axis_views(user, item, id_dim=4, axis_dim=2)

    assert tuple(views) == ("id_only", "id_n", "id_v", "full")
    assert views["id_only"][0].shape == (2, 4)
    assert views["id_n"][0].shape == (2, 6)
    torch.testing.assert_close(views["id_v"][0], torch.cat([user[:, :4], user[:, 6:]], 1))
    torch.testing.assert_close(views["full"][1], item)


def test_full_score_is_exact_sum_of_id_n_and_v_scores():
    user = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    item = torch.tensor([[2.0, 1.0, -1.0, 0.5]])

    components = diagnostic.component_pair_scores(
        user, item, torch.tensor([0]), torch.tensor([0]), id_dim=2, axis_dim=1
    )

    assert components["full"].item() == pytest.approx(
        components["id"].item()
        + components["activity"].item()
        + components["transaction_value"].item()
    )


def test_rank_transition_table_groups_truth_items_by_clv_segment():
    users = np.array([0, 1])
    segments = np.array(["저CLV", "고CLV"])
    truth = {0: np.array([10, 11]), 1: np.array([20])}
    m1_ranks = {0: {10: 3, 11: 15}, 1: {20: 55}}
    m2_ranks = {0: {10: 12, 11: 7}, 1: {20: 40}}

    table = diagnostic.rank_transition_table(
        users=users,
        segments=segments,
        truth=truth,
        reference_ranks=m1_ranks,
        model_ranks=m2_ranks,
    )

    lookup = table.set_index(["segment", "reference_bucket", "model_bucket"])
    assert lookup.at[("저CLV", "1-10", "11-20"), "truth_item_count"] == 1
    assert lookup.at[("저CLV", "11-20", "1-10"), "truth_item_count"] == 1
    assert lookup.at[("고CLV", ">50", "21-50"), "truth_item_count"] == 1
    assert table.groupby("segment")["share_within_segment"].sum().to_dict() == {
        "고CLV": pytest.approx(1.0),
        "저CLV": pytest.approx(1.0),
    }


def test_preflight_is_checkpoint_only_and_does_not_claim_significance(tmp_path):
    cfg = diagnostic.configure_checkpoint_diagnostic(out_dir=str(tmp_path))
    summary = diagnostic.preflight_summary(cfg)

    assert summary["training"] is False
    assert summary["views"] == ["id_only", "id_n", "id_v", "full"]
    assert summary["rank_buckets"] == ["1-10", "11-20", "21-50", ">50"]
    assert summary["statistical_note"] == "descriptive checkpoint diagnostic; no significance claim"
