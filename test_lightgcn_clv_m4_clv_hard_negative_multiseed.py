import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m4_clv_hard_negative_multiseed as multi


def test_protocol_is_frozen_to_ten_seeds_and_four_paired_arms(tmp_path):
    cfg = multi.configure_multiseed_run(
        out_dir=str(tmp_path / "multi"),
        baseline_result_dir=str(tmp_path / "baseline"),
        seed42_result_dir=str(tmp_path / "seed42"),
    )
    summary = multi.preflight_summary(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert summary["models"] == [
        "m1_64",
        "m1_multineg_mean_k5",
        "m4_clv_hard_k5",
        "m4_clv_hard_k5_degree_shuffled",
    ]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["minimum_positive_seed_count"] == 7


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("seeds", (42, 43)),
        ("epochs", 99),
        ("negative_count", 4),
        ("shuffle_degree_bins", 5),
        ("dataset", "hm"),
    ],
)
def test_protocol_rejects_unplanned_changes(tmp_path, override, value):
    with pytest.raises(ValueError, match="M4 10-seed"):
        multi.configure_multiseed_run(
            out_dir=str(tmp_path / "multi"),
            baseline_result_dir=str(tmp_path / "baseline"),
            seed42_result_dir=str(tmp_path / "seed42"),
            **{override: value},
        )


def test_degree_matched_shuffle_is_deterministic_and_preserves_each_stratum():
    q_clv = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    valid = np.ones(20, dtype=bool)
    degree = np.arange(1, 21, dtype=np.int64)

    first = multi.degree_matched_q_clv_shuffle(
        q_clv, valid, degree, n_bins=4, seed=1042
    )
    second = multi.degree_matched_q_clv_shuffle(
        q_clv, valid, degree, n_bins=4, seed=1042
    )

    np.testing.assert_array_equal(first["q_clv"], second["q_clv"])
    np.testing.assert_array_equal(first["stratum"], second["stratum"])
    assert first["changed_valid_user_share"] > 0.0
    np.testing.assert_allclose(np.sort(first["q_clv"]), np.sort(q_clv))
    for stratum in np.unique(first["stratum"][valid]):
        index = first["stratum"] == stratum
        np.testing.assert_allclose(
            np.sort(first["q_clv"][index]), np.sort(q_clv[index])
        )


def _metric_row(seed, model_id, increment):
    base = float(seed) / 1000.0
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


def test_decision_requires_positive_mean_and_seven_wins_against_both_controls():
    rows = []
    for seed in multi.SEEDS:
        rows.extend(
            [
                _metric_row(seed, multi.K1_MODEL_ID, 0.0000),
                _metric_row(seed, multi.MEAN_K5_MODEL_ID, 0.0010),
                _metric_row(seed, multi.M4_MODEL_ID, 0.0030),
                _metric_row(seed, multi.SHUFFLED_M4_MODEL_ID, 0.0020),
            ]
        )

    decision, paired = multi.multiseed_decision(pd.DataFrame(rows))

    assert decision["positive_screen"] is True
    assert set(paired["reference"]) == {
        multi.MEAN_K5_MODEL_ID,
        multi.SHUFFLED_M4_MODEL_ID,
    }
    assert paired["positive_seed_count"].eq(10).all()
    assert paired["mean_delta"].gt(0).all()

    failing = pd.DataFrame(rows)
    mask = (
        (failing.model_id == multi.M4_MODEL_ID)
        & (failing.seed >= 48)
    )
    failing.loc[mask, "고CLV_ndcg@10"] -= 0.01
    failed, _ = multi.multiseed_decision(failing)
    assert failed["positive_screen"] is False


def test_colab_runs_the_multiseed_runner_once_and_never_opens_test():
    notebook = json.loads(
        Path("clv_m4_clv_hard_negative_dunnhumby_multiseed_colab.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_multiseed(cfg)") == 1
    assert "configure_multiseed_run" in source
    assert "EVAL_TEST=True" not in source
    assert "holdout" not in source.lower()
