import json
from pathlib import Path

import pytest

import lightgcn_clv_m5_economic_positive_weight_test as final_test


def _config(tmp_path, **overrides):
    return final_test.configure_m5_economic_positive_test_run(
        out_dir=str(tmp_path / "results_m5_test"),
        **overrides,
    )


def test_default_run_is_one_seed_test_only_without_validation_or_holdout(tmp_path):
    cfg = _config(tmp_path)
    summary = final_test.preflight_summary(cfg)
    base = final_test._base_config(cfg)

    assert cfg.seeds == (42,)
    assert summary["planned_full_seeds"] == list(range(42, 52))
    assert summary["training_data"] == "DAY 1--697 (former train + validation)"
    assert summary["test_data"] == "DAY 698--704"
    assert summary["validation_constructed"] is False
    assert summary["validation_selection"] is False
    assert summary["early_stopping"] is False
    assert summary["post_test_rows"] == "DAY 705--711 ignored"
    assert summary["holdout_evaluation"] is False
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["HOLDOUT_DAYS"] == 7
    assert base["TIME_CUTOFF"] is None


def test_test_protocol_keeps_the_frozen_m5_architecture_and_six_arms(tmp_path):
    summary = final_test.preflight_summary(_config(tmp_path))

    assert summary["models"] == list(final_test.MODEL_IDS)
    assert summary["m2"]["rho"] == pytest.approx(0.15)
    assert summary["m2"]["economic_bins"] == 4
    assert summary["m2"]["shrinkage_strength"] == pytest.approx(10.0)
    assert summary["m4_prime"]["lambda"] == pytest.approx(0.5)
    assert summary["m4_prime"]["negative_count"] == 5
    assert summary["test_evaluation"] == (
        "one final-checkpoint evaluation per seed/model; completed results are cached"
    )


def test_full_run_requires_reusing_completed_seed42_result(tmp_path):
    with pytest.raises(ValueError, match="seed 42 결과 JSON"):
        _config(tmp_path, seeds=final_test.FULL_SEEDS)

    cfg = _config(
        tmp_path,
        seeds=final_test.FULL_SEEDS,
        reused_seed42_json="/content/drive/completed-seed42.json",
    )
    summary = final_test.preflight_summary(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert summary["current_scope"] == "frozen ten-seed final run"
    assert summary["seed42_handling"] == (
        "reuse the completed seed-42 test result; train seeds 43--51 only"
    )


def test_reused_seed42_loader_accepts_only_matching_complete_test_arms(tmp_path):
    source = tmp_path / "seed42.json"
    cfg = _config(
        tmp_path,
        seeds=final_test.FULL_SEEDS,
        reused_seed42_json=str(source),
    )
    manifest = {
        "transactions": {
            "path": "transactions.csv",
            "bytes": 1,
            "sha256": "transaction-hash",
        },
        "item_metadata": {
            "path": "products.csv",
            "bytes": 1,
            "sha256": "metadata-hash",
        },
    }
    arms = [
        {
            "seed": 42,
            "model_id": model_id,
            "split": "test",
            "test_evaluation_count": 1,
            "final_epoch": 100,
        }
        for model_id in final_test.MODEL_IDS
    ]
    source.write_text(
        json.dumps(
            {
                "code_version": final_test.CODE_VERSION,
                "source_revision": "pilot-revision",
                "config": {
                    **{
                        field: getattr(cfg, field)
                        for field in final_test._REUSE_CONFIG_FIELDS
                    },
                    "seeds": [42],
                },
                "input_manifest": manifest,
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )

    loaded = final_test._load_reused_seed42_arms(
        {"input_hash": final_test.moe.manifest_hash(manifest)}, cfg
    )

    assert len(loaded) == 6
    assert {arm["model_id"] for arm in loaded} == set(final_test.MODEL_IDS)
    assert all(
        arm["result_origin"] == "reused_completed_seed42_test" for arm in loaded
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("epochs", 99),
        ("rho", 0.05),
        ("positive_weight_lambda", 0.2),
        ("economic_bins", 3),
        ("shrinkage_strength", 0.0),
        ("negative_count", 1),
        ("n_layers", 1),
        ("seeds", (43,)),
    ],
)
def test_seed42_protocol_rejects_model_or_seed_changes(tmp_path, name, value):
    with pytest.raises(ValueError, match="M5 test-only"):
        _config(tmp_path, **{name: value})


def test_split_guard_rejects_validation_or_wrong_final_boundaries():
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
    final_test.validate_final_test_data(valid, "dunnhumby")

    contaminated = {
        **valid,
        "splits": {"val": (object(), object()), "test": (object(), object())},
    }
    with pytest.raises(RuntimeError, match="test split"):
        final_test.validate_final_test_data(contaminated, "dunnhumby")

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
        final_test.validate_final_test_data(wrong_boundary, "dunnhumby")


def test_hm_seed42_uses_full_two_year_test_only_protocol(tmp_path):
    cfg = _config(
        tmp_path,
        dataset="hm",
        batch_size=131_072,
    )
    summary = final_test.preflight_summary(cfg)
    base = final_test._base_config(cfg)

    assert cfg.seeds == (42,)
    assert summary["period"] == "full_history_about_2_years"
    assert summary["training_data"] == (
        "through 2020-09-08 (former train + validation)"
    )
    assert summary["test_data"] == "2020-09-09--15"
    assert summary["post_test_rows"] == "2020-09-16--22 ignored"
    assert summary["planned_full_seeds"] is None
    assert base["WINDOW_DAYS"] is None
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False


def test_hm_rejects_multiseed_scope(tmp_path):
    with pytest.raises(ValueError, match="H&M seed 42"):
        _config(
            tmp_path,
            dataset="hm",
            seeds=final_test.FULL_SEEDS,
            reused_seed42_json="not-allowed.json",
        )


def test_hm_split_guard_accepts_only_fixed_test_dates():
    valid = {
        "splits": {"test": (object(), object())},
        "data_stats": {
            "split_boundaries": {
                "train": {"end_inclusive": "2020-09-08T00:00:00"},
                "test": {
                    "start_exclusive": "2020-09-08T00:00:00",
                    "end_inclusive": "2020-09-15T00:00:00",
                },
                "holdout": {
                    "start_exclusive": "2020-09-15T00:00:00",
                    "end_inclusive": "2020-09-22T00:00:00",
                },
            },
            "split_evaluation_status": {
                "val": "merged_into_train",
                "test": "constructed",
                "holdout": "not_constructed",
            },
        },
    }
    final_test.validate_final_test_data(valid, "hm")

    valid["data_stats"]["split_boundaries"]["test"]["end_inclusive"] = (
        "2020-09-14T00:00:00"
    )
    with pytest.raises(RuntimeError, match="H&M 고정 test"):
        final_test.validate_final_test_data(valid, "hm")


def test_colab_starts_one_seed_test_only_run_once():
    notebook = json.loads(
        Path(
            "clv_m5_economic_positive_weight_dunnhumby_test_seed42_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_m5_economic_positive_test(cfg)") == 1
    assert "configure_m5_economic_positive_test_run" in source
    assert "215255f012907ff440388bfb190ac36d02873623" in source
    assert "TO_BE_PINNED" not in source
    assert "cfg.seeds == (42,)" in source
    assert "summary['validation_constructed'] is False" in source
    assert "summary['holdout_evaluation'] is False" in source


def test_multiseed_colab_reuses_seed42_and_runs_frozen_full_seeds_once():
    notebook = json.loads(
        Path(
            "clv_m5_economic_positive_weight_dunnhumby_test_multiseed_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_m5_economic_positive_test(cfg)") == 1
    assert "seeds=FULL_SEEDS" in source
    assert "reused_seed42_json=PILOT_RESULT" in source
    assert "m5_economic_positive_weight_test_b22a507c8ab5.json" in source
    assert "cfg.seeds == tuple(range(42, 52))" in source
    assert "train seeds 43--51 only" in source
    assert "summary['validation_constructed'] is False" in source
    assert "summary['holdout_evaluation'] is False" in source
    assert "28cf7b2ab8ad0f9e05a1ff8f0af02da378da845d" in source
    assert "TO_BE_PINNED" not in source
