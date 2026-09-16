import json
from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m4_k1_assignment_control_multiseed as multi


def _cfg(tmp_path):
    return multi.configure_m4_k1_assignment_control_multiseed(
        out_dir=str(tmp_path / "multi"),
        baseline_result_dir=str(tmp_path / "baseline"),
        seed42_result_dir=str(tmp_path / "seed42"),
    )


def test_contract_is_frozen_to_ten_development_seeds_and_three_k1_arms(tmp_path):
    cfg = _cfg(tmp_path)
    summary = multi.preflight_summary(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.negative_count == 1
    assert summary["trained_models_per_seed"] == list(multi.MODEL_IDS)
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("seeds", (42, 43)),
        ("negative_count", 5),
        ("epochs", 99),
        ("positive_weight_lambda", 0.25),
        ("time_cutoff", 704),
        ("minimum_positive_seed_count", 6),
    ],
)
def test_contract_rejects_unplanned_changes(tmp_path, override, value):
    with pytest.raises(ValueError, match="M4 K=1 10-seed"):
        multi.configure_m4_k1_assignment_control_multiseed(
            out_dir=str(tmp_path / "multi"),
            baseline_result_dir=str(tmp_path / "baseline"),
            seed42_result_dir=str(tmp_path / "seed42"),
            **{override: value},
        )


def test_each_training_seed_also_determines_its_degree_matched_shuffle(tmp_path):
    cfg = _cfg(tmp_path)

    seed42 = multi._single_config(cfg, seed=42)
    seed51 = multi._single_config(cfg, seed=51)

    assert seed42.seed == seed42.shuffle_seed == 42
    assert seed51.seed == seed51.shuffle_seed == 51
    assert seed42.negative_count == seed51.negative_count == 1


def _metric_row(seed, model_id, increment):
    base = 0.02 + (seed - 42) / 10000
    return {
        "seed": seed,
        "model_id": model_id,
        "role": model_id,
        "split": "historical_development_days_684_690",
        "final_epoch": 100,
        "m4_assignment": model_id,
        "recall@10": base + increment,
        "ndcg@10": base + increment,
        "recall@20": base + increment,
        "ndcg@20": base + increment,
        "recall@50": base + increment,
        "ndcg@50": base + increment,
        "price_purchase_amount_weighted_hit@10": base + increment,
        "vndcg@10": base + increment,
        "coverage@10": base,
        "user_value_tendency_recommended_price_alignment": base,
    }


def _passing_absolute():
    rows = []
    for seed in multi.FULL_SEEDS:
        rows.extend(
            [
                _metric_row(seed, multi.M1_MODEL_ID, 0.0000),
                _metric_row(seed, multi.M4_SHUFFLED_MODEL_ID, 0.0010),
                _metric_row(seed, multi.M4_ACTUAL_MODEL_ID, 0.0020),
            ]
        )
    return pd.DataFrame(rows)


def test_multiseed_decision_requires_positive_mean_seven_wins_and_accuracy_guard():
    absolute = _passing_absolute()
    comparison = multi.seedwise_comparison(absolute)
    paired = multi.paired_summary(comparison)

    passed = multi.multiseed_decision(absolute, paired)
    assert passed["multiseed_assignment_pass"] is True
    assert passed["actual_beats_m1_on_both_primary_metrics"] is True
    assert passed["actual_beats_shuffle_on_both_primary_metrics"] is True
    assert len(passed["primary_paired_results"]) == 4

    six_wins = _passing_absolute()
    mask = (
        (six_wins.model_id == multi.M4_ACTUAL_MODEL_ID)
        & (six_wins.seed >= 48)
    )
    six_wins.loc[
        mask, "price_purchase_amount_weighted_hit@10"
    ] -= 0.01
    paired = multi.paired_summary(multi.seedwise_comparison(six_wins))
    failed = multi.multiseed_decision(six_wins, paired)
    assert failed["actual_beats_shuffle_on_both_primary_metrics"] is False
    assert failed["multiseed_assignment_pass"] is False

    accuracy_fail = _passing_absolute()
    accuracy_fail.loc[
        accuracy_fail.model_id == multi.M4_ACTUAL_MODEL_ID, "recall@50"
    ] *= 0.85
    paired = multi.paired_summary(multi.seedwise_comparison(accuracy_fail))
    failed = multi.multiseed_decision(accuracy_fail, paired)
    assert failed["six_accuracy_mean_metrics_at_least_99pct_of_m1"] is False
    assert failed["multiseed_assignment_pass"] is False


def test_colab_runs_the_locked_multiseed_protocol_once():
    notebook_path = Path(
        "clv_m4_personalized_positive_weight_k1_assignment_control_"
        "multiseed_dunnhumby_colab.ipynb"
    )
    if not notebook_path.exists():
        return
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("run_m4_k1_assignment_control_multiseed(cfg)") == 1
    assert "cfg.seeds == tuple(range(42, 52))" in source
    assert "cfg.negative_count == 1" in source
    assert "final_test_constructed" in source
    assert "holdout_constructed" in source
    assert "TO_BE_PINNED" not in source
