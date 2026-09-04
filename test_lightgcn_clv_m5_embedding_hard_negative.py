import json
from pathlib import Path

import pytest

import lightgcn_clv_m5_embedding_hard_negative as m5


def test_preflight_freezes_minimal_m2_plus_m4_factorial(tmp_path):
    cfg = m5.configure_m5_run(
        out_dir=str(tmp_path / "m5"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    summary = m5.preflight_summary(cfg)

    assert summary["trained_models"] == list(m5.TRAINED_MODEL_IDS)
    assert summary["m2"]["rho"] == 0.05
    assert summary["m4"]["uniform_negative_count"] == 5
    assert summary["m4"]["per_positive_loss_mass"] == 1.0
    assert summary["attribution_control"]["same_permutation_in_m2_and_m4"] is True
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["m3_edge_weight"] is False
    assert summary["fixed"]["external_reranking"] is False
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 43),
        ("rho", 0.10),
        ("item_price_budget", 0.50),
        ("negative_count", 4),
        ("epochs", 99),
        ("shuffle_degree_bins", 5),
        ("dataset", "hm"),
    ],
)
def test_config_rejects_unplanned_changes(tmp_path, key, value):
    with pytest.raises(ValueError, match="빠른 M5 screen"):
        m5.configure_m5_run(
            out_dir=str(tmp_path / "m5"),
            baseline_result_dir=str(tmp_path / "baseline"),
            **{key: value},
        )


def _metrics(increment=0.0):
    return {
        "recall@10": 0.020 + increment,
        "ndcg@10": 0.021 + increment,
        "recall@20": 0.030 + increment,
        "ndcg@20": 0.031 + increment,
        "recall@50": 0.050 + increment,
        "ndcg@50": 0.041 + increment,
        "price_purchase_amount_weighted_hit@10": 0.380 + increment,
        "고CLV_recall@10": 0.010 + increment,
        "고CLV_ndcg@10": 0.011 + increment,
        "coverage@10": 0.005,
        "n_distinct@10": 450.0,
        "top10_share@10": 0.32,
    }


def _passing_metric_rows():
    return {
        m5.M1_K5_MODEL_ID: _metrics(0.0000),
        m5.M2_K5_MODEL_ID: _metrics(0.0002),
        m5.M4_MODEL_ID: _metrics(0.0010),
        m5.M5_MODEL_ID: _metrics(0.0020),
        m5.M5_SHUFFLED_MODEL_ID: _metrics(0.0015),
    }


def test_screen_requires_synergy_m4_gain_and_joint_clv_attribution():
    rows = _passing_metric_rows()
    decision = m5.screening_reading(rows)

    assert decision["positive_screen"] is True
    assert decision["interaction_pass"] is True
    assert decision["attribution_pass"] is True

    no_synergy = _passing_metric_rows()
    no_synergy[m5.M5_MODEL_ID]["price_purchase_amount_weighted_hit@10"] = 0.3811
    failed = m5.screening_reading(no_synergy)
    assert failed["interaction_pass"] is False
    assert failed["positive_screen"] is False

    shuffled_wins = _passing_metric_rows()
    shuffled_wins[m5.M5_SHUFFLED_MODEL_ID]["고CLV_ndcg@10"] = 0.02
    failed = m5.screening_reading(shuffled_wins)
    assert failed["high_clv_pass"] is False
    assert failed["positive_screen"] is False


def test_interaction_table_uses_factorial_difference_in_differences():
    table = m5.interaction_rows(_passing_metric_rows()).set_index("metric")
    value = table.loc[
        "price_purchase_amount_weighted_hit@10", "interaction_effect"
    ]
    assert value == pytest.approx(0.0008)


def test_colab_runs_one_seed42_screen_without_protected_evaluation():
    notebook_path = Path("clv_m5_m2_m4_joint_dunnhumby_colab.ipynb")
    if not notebook_path.exists():
        pytest.skip("notebook is added after the reviewed source commit")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert source.count("result_df = run_m5_screen(cfg)") == 1
    assert "configure_m5_run" in source
    assert "EVAL_TEST=True" not in source
    assert "EVAL_HOLDOUT=True" not in source
    assert "PIN_AFTER_LOCAL_COMMIT" not in source
