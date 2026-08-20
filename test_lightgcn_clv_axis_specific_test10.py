import json
from pathlib import Path

import pytest

import lightgcn_clv_axis_specific_test10 as final10


def test_final_protocol_is_locked_to_ten_paired_seeds_and_four_models(tmp_path):
    cfg = final10.configure_test10_run(out_dir=str(tmp_path))
    summary = final10.preflight_summary(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.epochs == 100
    assert summary["models"] == [
        "m1_64",
        "m2_axis_specific_gate",
        "m1_96",
        "m2_shuffled_user",
    ]
    assert summary["validation_selection"] is False
    assert summary["early_stopping"] is False
    assert summary["automatic_epoch_resume"] is True


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("seeds", (42, 43)),
        ("epochs", 99),
        ("dataset", "hm"),
        ("gate_shape", "equal"),
        ("axis_dim", 8),
    ],
)
def test_final_protocol_rejects_unapproved_changes(tmp_path, override, value):
    with pytest.raises(ValueError):
        final10.configure_test10_run(out_dir=str(tmp_path), **{override: value})


def test_base_config_merges_validation_and_opens_only_test(tmp_path):
    cfg = final10.configure_test10_run(out_dir=str(tmp_path))
    base = final10._base_config(cfg)

    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_USER_INTER"] == 1
    assert base["MIN_ITEM_INTER"] == 1
    assert base["EPOCHS"] == 100


def test_four_model_specs_include_capacity_and_assignment_controls():
    assert final10._model_spec("m1_64") == ("m1", 64)
    assert final10._model_spec("m2_axis_specific_gate") == ("joint_nv", 96)
    assert final10._model_spec("m1_96") == ("m1", 96)
    assert final10._model_spec("m2_shuffled_user") == (
        "joint_shuffled_user",
        96,
    )


def test_ten_seed_reporting_keeps_each_seed_and_computes_paired_means():
    arms = []
    empty_diagnostics = {
        "activity_axis_weight": None,
        "transaction_value_axis_weight": None,
        "activity_gate_mean": None,
        "activity_gate_std": None,
        "transaction_value_gate_mean": None,
        "transaction_value_gate_std": None,
    }
    for seed in final10.SEEDS:
        for model_id in final10.MODELS:
            baseline = float(seed)
            increment = 1.0 if model_id == "m2_axis_specific_gate" else 0.0
            arms.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "role": "test",
                    "final_epoch": 100,
                    "diagnostics": empty_diagnostics,
                    "metrics": {"recall@10": baseline + increment},
                }
            )

    absolute = final10._absolute_rows(arms)
    absolute_mean, paired_seed, paired_mean = final10._summary_tables(
        absolute, arms
    )

    assert len(absolute) == 40
    assert absolute.groupby("seed").size().eq(4).all()
    m2_absolute = absolute_mean.query(
        "model_id == 'm2_axis_specific_gate' and metric == 'recall@10'"
    ).iloc[0]
    assert m2_absolute["mean"] == pytest.approx(47.5)
    m2_seed_delta = paired_seed.query(
        "model_id == 'm2_axis_specific_gate' and metric == 'recall@10'"
    )
    assert len(m2_seed_delta) == 10
    assert m2_seed_delta["delta"].eq(1.0).all()
    m2_mean_delta = paired_mean.query(
        "model_id == 'm2_axis_specific_gate' and metric == 'recall@10'"
    ).iloc[0]
    assert m2_mean_delta["mean"] == pytest.approx(1.0)
    assert m2_mean_delta["positive_seed_count"] == 10


def test_public_results_use_professor_facing_axis_and_metric_terms():
    metrics = final10._public_metrics(
        {"revenue@10": 1.0, "arp@10": 0.2, "value_alignment": 0.1}
    )

    assert "price_purchase_amount_weighted_hit@10" in metrics
    assert "mean_recommended_price_percentile@10" in metrics
    assert "user_value_tendency_recommended_price_alignment" in metrics
    summary = final10.preflight_summary(
        final10.configure_test10_run(out_dir="/tmp/final-test10")
    )
    assert "activity_axis_weight" in summary["m2"]
    assert "transaction_value_axis_weight" in summary["m2"]


def test_colab_is_pinned_and_starts_the_ten_seed_runner_once():
    notebook = json.loads(
        Path("clv_m2_axis_specific_gate_dunnhumby_test10_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "TO_BE_PINNED_AFTER_REVIEW" not in source
    assert "60ea811a1cc8b191f5a2f54d502e171846d97f3f" in source
    assert source.count("result_df = run_test10(cfg)") == 1
    assert "configure_test10_run" in source
    assert "progress.json" in source
    assert "ACKNOWLEDGE_HIGH_COST" not in source
