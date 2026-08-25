import pandas as pd
import pytest

import lightgcn_clv_gatefree_lowdim_independent_dropout_diagnostic as diagnostic


def _view_row(view, base):
    row = {"view": view}
    for segment in ("", "저CLV_", "중CLV_", "고CLV_"):
        for cutoff in (10, 20, 50):
            row[f"{segment}recall@{cutoff}"] = base
            row[f"{segment}ndcg@{cutoff}"] = base + 0.1
            row[f"{segment}price_purchase_amount_weighted_hit@{cutoff}"] = base + 0.2
    return row


def test_axis_effect_table_uses_id_only_as_axis_reference():
    frame = pd.DataFrame(
        [
            _view_row("m1_64", 1.0),
            _view_row("id_only", 2.0),
            _view_row("id_n", 2.3),
            _view_row("id_v", 2.5),
            _view_row("full", 2.7),
        ]
    )

    table = diagnostic.axis_effect_table(frame)
    row = table[(table.segment == "고CLV") & (table.metric == "recall@10")].iloc[0]

    assert row["id_n_delta_vs_id_only"] == pytest.approx(0.3)
    assert row["id_v_delta_vs_id_only"] == pytest.approx(0.5)
    assert row["full_delta_vs_m1"] == pytest.approx(1.7)


def test_rank_flow_summary_counts_truth_items_entering_and_leaving_top10():
    transition = pd.DataFrame(
        [
            {
                "segment": "저CLV",
                "reference_bucket": "1-10",
                "model_bucket": "11-20",
                "truth_item_count": 3,
            },
            {
                "segment": "저CLV",
                "reference_bucket": "11-20",
                "model_bucket": "1-10",
                "truth_item_count": 5,
            },
            {
                "segment": "저CLV",
                "reference_bucket": "1-10",
                "model_bucket": "1-10",
                "truth_item_count": 7,
            },
        ]
    )

    table = diagnostic.rank_flow_summary(
        transition, reference="id_only", view="id_v"
    )
    top10 = table[(table.segment == "저CLV") & (table.cutoff == 10)].iloc[0]

    assert top10["entered_topk"] == 5
    assert top10["exited_topk"] == 3
    assert top10["net_truth_items"] == 2
    assert top10["retained_topk"] == 7


def test_preflight_is_checkpoint_only(tmp_path):
    cfg = diagnostic.configure_independent_dropout_diagnostic(out_dir=str(tmp_path))
    summary = diagnostic.preflight_summary(cfg)

    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["views"] == ["m1_64", "id_only", "id_n", "id_v", "full"]
