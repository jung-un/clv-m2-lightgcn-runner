import json
from pathlib import Path

import pytest

import lightgcn_clv_level_composition_price_test1 as final_test


def _config(tmp_path, **overrides):
    return final_test.configure_m2_level_composition_price_test_run(
        out_dir=str(tmp_path / "results_m2_test"),
        **overrides,
    )


def test_default_run_is_seed42_test_only_without_validation_or_holdout(tmp_path):
    cfg = _config(tmp_path)
    summary = final_test.preflight_summary(cfg)
    base = final_test._base_config(cfg)

    assert cfg.seed == 42
    assert summary["training_data"] == "DAY 1--697 (former train + validation)"
    assert summary["test_data"] == "DAY 698--704"
    assert summary["validation_constructed"] is False
    assert summary["validation_selection"] is False
    assert summary["early_stopping"] is False
    assert summary["post_test_rows"] == "DAY 705--711 ignored"
    assert summary["holdout_constructed"] is False
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["HOLDOUT_DAYS"] == 7
    assert base["TIME_CUTOFF"] is None


def test_protocol_keeps_the_selected_m2_and_paired_controls(tmp_path):
    summary = final_test.preflight_summary(_config(tmp_path))

    assert summary["trained_models"] == list(final_test.TRAINED_MODELS)
    assert summary["reported_models"] == list(final_test.REPORTED_MODELS)
    assert summary["m2"]["architecture"] == (
        "ID(64)|CLV level/composition relation(2)|explicit price fit(1)"
    )
    assert summary["m2"]["rho"] == pytest.approx(0.05)
    assert summary["m2"]["item_price_budget"] == pytest.approx(0.25)
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["paired_controls"]["same_seed_initialization_batches_and_negatives"]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("seed", 43),
        ("epochs", 99),
        ("id_dim", 32),
        ("clv_dim", 4),
        ("rho", 0.025),
        ("item_price_budget", 0.5),
        ("n_layers", 1),
        ("batch_size", 4096),
        ("lr", 1e-3),
        ("pref_reg", 0.0),
        ("shuffle_seed", 42),
    ],
)
def test_seed42_protocol_rejects_model_or_training_changes(
    tmp_path, name, value
):
    with pytest.raises(ValueError, match="M2 seed-42 test-only"):
        _config(tmp_path, **{name: value})


def test_split_guard_rejects_validation_or_wrong_boundaries():
    valid = {
        "splits": {"test": (object(), object())},
        "data_stats": {
            "split_boundaries": {
                "train": {"end_inclusive": 697.0},
                "test": {"start_exclusive": 697.0, "end_inclusive": 704.0},
                "holdout": {"start_exclusive": 704.0, "end_inclusive": 711.0},
            },
            "split_evaluation_status": {
                "val": "merged_into_train",
                "test": "constructed",
                "holdout": "not_constructed",
            },
        },
    }
    final_test.validate_final_test_data(valid)

    contaminated = {
        **valid,
        "splits": {"val": (object(), object()), "test": (object(), object())},
    }
    with pytest.raises(RuntimeError, match="test split"):
        final_test.validate_final_test_data(contaminated)

    wrong_boundary = {
        **valid,
        "data_stats": {
            **valid["data_stats"],
            "split_boundaries": {
                **valid["data_stats"]["split_boundaries"],
                "test": {"start_exclusive": 704.0, "end_inclusive": 711.0},
            },
        },
    }
    with pytest.raises(RuntimeError, match="DAY 698--704"):
        final_test.validate_final_test_data(wrong_boundary)


def test_descriptive_reading_never_makes_a_single_seed_final_decision():
    def metrics(scale=1.0):
        values = {name: scale for name in final_test.ACCURACY_METRICS}
        values.update(
            {
                "price_purchase_amount_weighted_hit@10": scale,
                "고CLV_recall@10": scale,
                "고CLV_ndcg@10": scale,
            }
        )
        return values

    arms = {
        final_test.MATCHED_MODEL_ID: {"metrics": metrics(1.0)},
        final_test.MODEL_ID: {"metrics": metrics(1.1)},
        final_test.SHUFFLED_MODEL_ID: {"metrics": metrics(1.05)},
        final_test.ID_ONLY_MODEL_ID: {"metrics": metrics(1.02)},
    }
    reading = final_test.descriptive_reading(arms)

    assert reading["descriptive_only"] is True
    assert reading["single_seed_final_decision_permitted"] is False


def test_colab_starts_the_test_only_run_once():
    notebook = json.loads(
        Path(
            "clv_m2_level_composition_price_dunnhumby_test_seed42_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count(
        "result_df = run_m2_level_composition_price_test(cfg)"
    ) == 1
    assert "configure_m2_level_composition_price_test_run" in source
    assert "TO_BE_PINNED" not in source
    assert "cfg.seed == 42" in source
    assert "summary['validation_constructed'] is False" in source
    assert "summary['holdout_constructed'] is False" in source
    assert "os.chdir('/content')" in source
