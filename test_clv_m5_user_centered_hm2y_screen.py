from dataclasses import asdict
from pathlib import Path
import tempfile

import pandas as pd
import pytest

import clv_m5_user_centered_hm2y_screen as s


def _curve(shuffle_gain=0.01):
    rows = []
    for epoch in (100, 200, 300):
        for model, factor in (
            (s.M1, 1.0),
            (s.M3, 1.0),
            (s.M5_C_SHUFFLED, 1.01),
            (s.M5_C, 1.01 + shuffle_gain),
        ):
            rows.append({
                "model_id": model,
                "seed": 43,
                "epoch": epoch,
                **{metric: factor for metric in (*s.ACCURACY, *s.ECONOMIC)},
            })
    return pd.DataFrame(rows)


def test_hm_screen_is_development_only_and_fixed_seed43():
    cfg = s.configure(out_dir="/tmp/hm")
    assert cfg.seed == 43
    assert cfg.epochs == 300
    assert cfg.evaluation_epochs == (100, 200, 300)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert s.models_for(cfg) == (s.M1, s.M3, s.M5_C, s.M5_C_SHUFFLED)

    with pytest.raises(ValueError):
        s.configure(seed=44)
    with pytest.raises(ValueError):
        s.configure(eval_test=True)


def test_hm_reading_is_transfer_screen_not_final_claim():
    cfg = s.configure(out_dir="/tmp/hm")
    reading = s.reading(_curve(), cfg)

    assert reading["development_transfer_screen_only"]
    assert reading["final_test_evaluated"] is False
    assert reading["m5_c_accuracy_guard_99pct_vs_m1_and_m3"]
    assert reading["m5_c_both_economic_at10_above_m1_and_m3"]
    assert reading["observed_assignment_beats_shuffle_on_both_economic_at10"]
    assert reading["eligible_for_future_final_evaluation"]
    assert reading["success_claim"] is False

    tied = s.reading(_curve(shuffle_gain=s.NUMERIC_TOLERANCE / 100), cfg)
    assert not tied["observed_assignment_beats_shuffle_on_both_economic_at10"]
    assert not tied["eligible_for_future_final_evaluation"]


def test_hm_report_requires_all_four_models_at_all_three_epochs():
    with tempfile.TemporaryDirectory() as directory:
        cfg = s.configure(out_dir=directory)
        prepared = {"run_dir": Path(directory), "protocol": {"config": asdict(cfg)}}
        curve = _curve()
        result = s.report(prepared, cfg, curve, diagnostics=pd.DataFrame())
        assert len(result["absolute"]) == 12
        assert all(Path(path).is_file() for path in result["paths"].values())
        with pytest.raises(RuntimeError):
            s.report(prepared, cfg, curve.iloc[:-1], diagnostics=pd.DataFrame())
