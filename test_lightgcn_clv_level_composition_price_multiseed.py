import json
from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_level_composition_price_multiseed as multi


def test_protocol_is_frozen_to_rho05_ten_seeds_and_four_reported_views(tmp_path):
    cfg = multi.configure_multiseed_run(
        out_dir=str(tmp_path / "multi"),
        baseline_result_dir=str(tmp_path / "baseline"),
        seed42_result_dir=str(tmp_path / "seed42"),
    )
    summary = multi.preflight_summary(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.rho == 0.05
    assert summary["models"] == [
        "m1_matched_rho0",
        "m2_clv_level_composition_price_embedding",
        "m2_degree_matched_clv_shuffle",
        "m2_jointly_trained_id_only",
    ]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("seeds", (42, 43)),
        ("rho", 0.10),
        ("epochs", 99),
        ("item_price_budget", 0.50),
        ("shuffle_degree_bins", 5),
        ("dataset", "hm"),
    ],
)
def test_protocol_rejects_unplanned_changes(tmp_path, override, value):
    with pytest.raises(ValueError, match="M2 10-seed"):
        multi.configure_multiseed_run(
            out_dir=str(tmp_path / "multi"),
            baseline_result_dir=str(tmp_path / "baseline"),
            seed42_result_dir=str(tmp_path / "seed42"),
            **{override: value},
        )


def _metric_row(seed, model_id, increment):
    base = 0.05 + float(seed - 42) / 1000.0
    return {
        "seed": seed,
        "model_id": model_id,
        "recall@10": base + increment,
        "ndcg@10": base + increment,
        "recall@20": base + increment,
        "ndcg@20": base + increment,
        "recall@50": base + increment,
        "ndcg@50": base + increment,
        "고CLV_recall@10": base + increment,
        "고CLV_ndcg@10": base + increment,
        "price_purchase_amount_weighted_hit@10": base + increment,
    }


def _passing_rows():
    rows = []
    for seed in multi.SEEDS:
        rows.extend(
            [
                _metric_row(seed, multi.MATCHED_MODEL_ID, 0.0000),
                _metric_row(seed, multi.ID_ONLY_MODEL_ID, 0.0010),
                _metric_row(seed, multi.SHUFFLED_MODEL_ID, 0.0015),
                _metric_row(seed, multi.MODEL_ID, 0.0030),
            ]
        )
    return rows


def test_decision_requires_overall_guard_and_seven_paired_wins_for_clv_attribution():
    decision, paired = multi.multiseed_decision(pd.DataFrame(_passing_rows()))

    assert decision["positive_screen"] is True
    assert decision["all_overall_metrics_not_below_matched"] is True
    assert paired["passes"].all()
    assert paired["positive_seed_count"].eq(10).all()

    six_wins = pd.DataFrame(_passing_rows())
    mask = (
        (six_wins.model_id == multi.MODEL_ID)
        & (six_wins.seed >= 48)
    )
    six_wins.loc[mask, "고CLV_recall@10"] -= 0.01
    failed, failed_paired = multi.multiseed_decision(six_wins)
    row = failed_paired[
        (failed_paired.reference == multi.ID_ONLY_MODEL_ID)
        & (failed_paired.metric == "고CLV_recall@10")
    ].iloc[0]
    assert row.positive_seed_count == 6
    assert not bool(row.passes)
    assert failed["positive_screen"] is False

    lower_overall = pd.DataFrame(_passing_rows())
    lower_overall.loc[
        lower_overall.model_id == multi.MODEL_ID, "recall@20"
    ] -= 0.004
    failed, _ = multi.multiseed_decision(lower_overall)
    assert failed["all_overall_metrics_not_below_matched"] is False
    assert failed["positive_screen"] is False


def test_colab_runs_multiseed_once_without_final_test_or_holdout():
    notebook = json.loads(
        Path("clv_m2_level_composition_price_rho05_multiseed_dunnhumby_colab.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_multiseed(cfg)") == 1
    assert "configure_multiseed_run" in source
    assert "PIN_AFTER_LOCAL_COMMIT" not in source
    assert "EVAL_HOLDOUT=True" not in source
    assert "EVAL_TEST=True" not in source
