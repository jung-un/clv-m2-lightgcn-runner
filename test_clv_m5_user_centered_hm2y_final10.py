from dataclasses import asdict
from pathlib import Path
import tempfile

import pandas as pd
import pytest

import clv_m5_user_centered_hm2y_final10 as s


def _arm_row(model_id, seed, factor=1.0):
    return {
        "model_id": model_id,
        "seed": seed,
        "metrics": {
            metric: factor * (seed - 40)
            for metric in (*s.ACCURACY, *s.ECONOMIC)
        },
        "diagnostics": {
            "graph": "binary",
            "row_weighted": False,
            "graph_audit": None,
            "weight_audit": None,
            "shuffle_audit": None,
        },
    }


def _absolute(seeds):
    rows = []
    for seed in seeds:
        for model, factor in (
            (s.M1, 1.00),
            (s.M3, 1.01),
            (s.M4_C, 1.01),
            (s.M5_C_SHUFFLED, 1.02),
            (s.M5_C, 1.03),
        ):
            rows.append({
                "model_id": model,
                "seed": seed,
                **{
                    metric: factor * (seed - 40)
                    for metric in (*s.ACCURACY, *s.ECONOMIC)
                },
            })
    return pd.DataFrame(rows)


def test_protocol_is_fixed_hm_last_week_without_validation_or_holdout():
    cfg = s.configure(out_dir="/tmp/hm-final")
    base = s.base_config(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.epochs == 300
    assert s.INITIAL_SEEDS == (49, 50)
    assert s.CORE_MODELS == (s.M1, s.M3, s.M5_C)
    assert s.models_for(cfg) == s.ALL_MODELS
    assert (base["VAL_DAYS"], base["TEST_DAYS"], base["HOLDOUT_DAYS"]) == (0, 7, 0)
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["MIN_ITEM_INTER"] == 1
    assert base["EARLY_STOP"] == cfg.epochs

    with pytest.raises(ValueError):
        s.configure(seeds=(49, 50))
    with pytest.raises(ValueError):
        s.configure(epochs=200)


def test_batch_validation_only_accepts_frozen_seed_and_model_subsets():
    s._validate_batch(s.INITIAL_SEEDS, s.CORE_MODELS)

    with pytest.raises(ValueError):
        s._validate_batch((52,), s.CORE_MODELS)
    with pytest.raises(ValueError):
        s._validate_batch(s.INITIAL_SEEDS, ("new_model",))


def test_initial_batch_is_partial_and_cannot_create_final_decision():
    rows = [
        _arm_row(model, seed)
        for seed in s.INITIAL_SEEDS
        for model in s.CORE_MODELS
    ]
    status = s.partial_status(rows)

    assert status["completed_arm_count"] == 6
    assert status["planned_arm_count"] == 50
    assert status["initial_batch_complete"] is True
    assert status["test_results_partial"] is True
    assert status["final_decision_available"] is False
    assert status["retuning_from_partial_results_allowed"] is False


def test_final_reading_requires_all_fifty_arms():
    cfg = s.configure(out_dir="/tmp/hm-final")
    absolute = _absolute(cfg.seeds)
    decision = s.reading(absolute, cfg)

    assert len(absolute) == 50
    assert decision["final_ten_seed_report"] is True
    assert decision["m5_c_accuracy_guard_99pct_vs_m1_and_m3_mean"] is True
    assert decision["m5_c_both_economic_at10_above_m1_and_m3_mean"] is True
    assert decision["observed_assignment_beats_shuffle_on_both_economic_at10_mean"] is True
    assert decision["final_condition_met"] is True
    assert decision["significance_claim"] is False

    with pytest.raises(RuntimeError):
        s.reading(absolute.iloc[:-1], cfg)


def test_partial_report_is_labelled_partial_and_writes_only_partial_artifacts():
    rows = [
        _arm_row(model, seed)
        for seed in s.INITIAL_SEEDS
        for model in s.CORE_MODELS
    ]
    with tempfile.TemporaryDirectory() as directory:
        prepared = {
            "run_dir": Path(directory),
            "protocol": {"config": asdict(s.configure(out_dir=directory))},
        }
        result = s.partial_report(prepared, rows)

        assert result["status"]["final_decision_available"] is False
        assert len(result["absolute"]) == 6
        assert all(Path(path).is_file() for path in result["paths"].values())
