import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m3_dunnhumby_test1 as t1


def _cfg(**overrides):
    return t1.configure_dunnhumby_test1(**({"out_dir": "/tmp/dh_test1"} | overrides))


def test_the_protocol_is_fixed_and_leaves_the_backtest():
    summary = t1.preflight_summary(_cfg())

    assert summary["split"] == t1.SPLIT == "dunnhumby_test_day_698_704"
    assert summary["backtest_cutoff_removed"] is True
    assert summary["never_touched"] == "DAY 705-711"
    assert summary["epochs_fixed"] == 300
    assert "no epoch, arm or checkpoint is selected" in summary["no_validation_selection"]

    for bad in ({"epochs": 100}, {"negative_count": 5}, {"dataset": "hm"},
                {"seeds": ()}, {"seeds": (42, 42)}):
        with pytest.raises(ValueError):
            _cfg(**bad)


def test_the_run_configuration_is_the_final_test_one():
    cfg = _cfg()
    base = t1._base_config(cfg, 42)
    assert base["TIME_CUTOFF"] is None          # 백테스트 컷오프를 벗어난다
    assert base["TRAIN_ON_VAL"] is True         # 검증 구간을 학습에 합친다
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False        # 705-711은 끝까지 닫아둔다
    assert base["EPOCHS"] == base["EARLY_STOP"] == 300


def _rows(m1, m3, seeds=(42, 43, 44)):
    metrics = list(t1.ACCURACY_METRICS) + list(t1.ECONOMIC_METRICS)
    out = []
    for seed, (a, b) in zip(seeds, zip(m1, m3)):
        out.append({"seed": seed, "model_id": t1.M1_MODEL_ID,
                    **{m: a for m in metrics}})
        out.append({"seed": seed, "model_id": t1.M3_MODEL_ID,
                    **{m: b for m in metrics}})
    return pd.DataFrame(out)


def test_the_reading_averages_over_seeds_and_counts_them():
    table = _rows([0.010, 0.010, 0.010], [0.011, 0.009, 0.012])
    reading = t1.averaged_reading(table, _cfg(seeds=(42, 43, 44)))

    assert reading["seed_count"] == 3 and reading["seeds"] == [42, 43, 44]
    one = reading["per_metric"]["recall@10"]
    assert one["m3_mean"] == pytest.approx(0.0106667, rel=1e-4)
    assert one["seeds_improved"] == 2            # 평균은 올라도 한 시드는 내려갔다
    assert one["paired_diff_sd"] > 0
    assert reading["condition_met"] is True
    assert reading["significance_claimed"] is False


def test_a_mean_that_fails_the_accuracy_guard_fails_the_condition():
    table = _rows([0.010, 0.010], [0.0098, 0.0098], seeds=(42, 43))
    reading = t1.averaged_reading(table, _cfg(seeds=(42, 43)))

    # 0.0098 / 0.010 = 0.98 이므로 경제지표도 내려가고 보호선도 깨진다
    assert reading["both_economic_metrics_improved_on_the_mean"] is False
    assert reading["six_accuracy_metrics_at_least_99pct_of_m1_on_the_mean"] is False
    assert reading["condition_met"] is False
    assert reading["worst_accuracy_ratio_on_the_mean"] == pytest.approx(0.98)


def test_a_seed_with_only_one_finished_arm_is_left_out():
    table = _rows([0.010, 0.010], [0.011, 0.011], seeds=(42, 43))
    table = table[~((table.seed == 43) & (table.model_id == t1.M3_MODEL_ID))]

    reading = t1.averaged_reading(table, _cfg(seeds=(42, 43)))
    assert reading["seeds"] == [42] and reading["seed_count"] == 1
    assert np.isnan(reading["per_metric"]["recall@10"]["paired_diff_sd"])

    with pytest.raises(RuntimeError, match="끝난 시드가 없습니다"):
        t1.averaged_reading(table[table.model_id.eq(t1.M1_MODEL_ID)], _cfg())
