from dataclasses import asdict
from pathlib import Path
import tempfile

import pandas as pd
import pytest

import clv_m5_user_centered_final10 as s


def _rows(seeds, shuffled_gain=0.01):
    rows = []
    for seed in seeds:
        for model, factor in (
            (s.M1, 1.0),
            (s.M3, 1.01),
            (s.M5_C_SHUFFLED, 1.02),
            (s.M5_C, 1.02 + shuffled_gain),
        ):
            rows.append({
                "model_id": model,
                "seed": seed,
                **{metric: factor * (seed - 40) for metric in (*s.ACCURACY, *s.ECONOMIC)},
            })
    return pd.DataFrame(rows)


def test_final_config_is_last_week_fixed_ten_seed_protocol():
    cfg = s.configure(out_dir="/tmp/final")
    base = s.base_config(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.epochs == 300
    assert (base["VAL_DAYS"], base["TEST_DAYS"], base["HOLDOUT_DAYS"]) == (0, 7, 0)
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["MIN_ITEM_INTER"] == 1
    assert s.models_for(cfg) == (s.M1, s.M3, s.M5_C, s.M5_C_SHUFFLED)

    with pytest.raises(ValueError):
        s.configure(seeds=(42,))
    with pytest.raises(ValueError):
        s.configure(epochs=200)


def test_final_reading_requires_all_seeds_and_uses_numerical_tolerance():
    cfg = s.configure(out_dir="/tmp/final")
    absolute = _rows(cfg.seeds)
    reading = s.reading(absolute, cfg)

    assert reading["final_ten_seed_report"]
    assert reading["m5_c_accuracy_guard_99pct_vs_m1_and_m3_mean"]
    assert reading["m5_c_both_economic_at10_above_m1_and_m3_mean"]
    assert reading["observed_assignment_beats_shuffle_on_both_economic_at10_mean"]
    assert reading["final_condition_met"]

    almost_tied = _rows(cfg.seeds, shuffled_gain=s.NUMERIC_TOLERANCE / 100)
    tied = s.reading(almost_tied, cfg)
    assert not tied["observed_assignment_beats_shuffle_on_both_economic_at10_mean"]
    assert not tied["final_condition_met"]

    with pytest.raises(RuntimeError):
        s.reading(absolute.iloc[:-1], cfg)


def test_report_writes_full_tables_without_significance_claim():
    with tempfile.TemporaryDirectory() as directory:
        cfg = s.configure(out_dir=directory)
        absolute = _rows(cfg.seeds)
        prepared = {"run_dir": Path(directory), "protocol": {"config": asdict(cfg)}}
        result = s.report(prepared, cfg, absolute, diagnostics=pd.DataFrame())

        assert result["summary"]["count"].eq(10).all()
        assert result["reading"]["significance_claim"] is False
        assert all(Path(path).is_file() for path in result["paths"].values())
