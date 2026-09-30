import pandas as pd
import pytest

import lightgcn_clv_m4_hm2y_confirmatory as conf
import lightgcn_clv_m4_k1_assignment_control_hm2y as screen


def _cfg(**overrides):
    defaults = {"out_dir": "/tmp/conf", "dev_out_dir": "/tmp/dev"}
    return conf.configure_confirmatory(**(defaults | overrides))


def test_the_protected_split_needs_an_explicit_decision():
    """Opening the test split is one-way, so it cannot happen by default."""

    cfg = _cfg()
    assert cfg.open_test_split is False
    with pytest.raises(RuntimeError, match="한 번만"):
        conf.run_confirmatory(cfg)


def test_everything_reported_is_fixed_before_the_run():
    summary = conf.preflight_summary(_cfg())

    assert summary["trains_nothing"] is True
    assert summary["protected_split_opened"] == conf.TEST_SPLIT
    assert summary["seeds"] == [42, 43]
    assert summary["arms"] == list(screen.MODEL_IDS)
    assert summary["primary_metrics"] == list(screen.ECONOMIC_METRICS)
    # 판정 조건은 동결된 개발 스크린에서 그대로 가져온다
    assert conf.ACCURACY_GUARD == screen.ACCURACY_GUARD
    assert conf.ACCURACY_METRICS == screen.ACCURACY_METRICS


def _rows(actual, shuffle, m1, seeds=(42, 43)):
    """One row per arm per seed, with every reported metric set to a scale."""

    metrics = list(screen.ACCURACY_METRICS) + list(screen.ECONOMIC_METRICS)
    rows = []
    for seed in seeds:
        for model_id, scale in ((screen.M1_MODEL_ID, m1),
                                (screen.M4_ACTUAL_MODEL_ID, actual),
                                (screen.M4_SHUFFLED_MODEL_ID, shuffle)):
            rows.append({"seed": seed, "model_id": model_id,
                         **{m: 0.01 * scale for m in metrics}})
    return pd.DataFrame(rows)


def test_a_seed_passes_only_when_it_beats_both_references():
    reading = conf.confirmatory_reading(_rows(actual=1.05, shuffle=1.02, m1=1.00))
    assert reading["seeds_passing"] == 2 and reading["seeds_evaluated"] == 2
    assert reading["per_seed"][42]["attribution_pass"] is True

    # 기준모형은 이기지만 순열을 못 이기면 귀속이 성립하지 않는다
    reading = conf.confirmatory_reading(_rows(actual=1.02, shuffle=1.05, m1=1.00))
    assert reading["seeds_passing"] == 0
    assert reading["per_seed"][42]["actual_beats_m1_on_both_economic_metrics"] is True
    assert reading["per_seed"][42]["actual_beats_shuffle_on_both_economic_metrics"] is False


def test_the_accuracy_guard_can_fail_a_seed_on_its_own():
    rows = _rows(actual=1.05, shuffle=1.02, m1=1.00)
    worst = screen.ACCURACY_METRICS[0]
    rows.loc[rows.model_id.eq(screen.M4_ACTUAL_MODEL_ID), worst] = 0.01 * 0.90

    reading = conf.confirmatory_reading(rows)
    seed = reading["per_seed"][42]
    assert seed["six_accuracy_metrics_at_least_99pct_of_m1"] is False
    assert seed["attribution_pass"] is False
    assert seed["worst_accuracy_ratio_vs_m1"] == pytest.approx(0.90)


def test_the_reading_never_claims_significance_or_post_hoc_selection():
    reading = conf.confirmatory_reading(_rows(actual=1.05, shuffle=1.02, m1=1.00))
    assert reading["significance_claimed"] is False
    assert reading["selected_after_seeing"] is False
    assert reading["split"] == conf.TEST_SPLIT


def test_the_arm_folder_is_found_not_recomputed(tmp_path):
    """The development hash includes the source revision, so it cannot be recomputed."""

    root = tmp_path / "arms" / "7218001fef1d"
    root.mkdir(parents=True)
    for model_id in screen.MODEL_IDS:
        (root / f"{model_id}_s43.json").write_text("{}")
        (root / f"{model_id}_s43.pt").write_bytes(b"x")
    assert conf.discover_arm_hash(str(tmp_path), 43) == "7218001fef1d"

    with pytest.raises(RuntimeError, match="결과 파일이 0개"):
        conf.discover_arm_hash(str(tmp_path), 42)

    other = tmp_path / "arms" / "429a3169f0d7"
    other.mkdir(parents=True)
    (other / f"{screen.M1_MODEL_ID}_s43.json").write_text("{}")
    with pytest.raises(RuntimeError, match="2개"):
        conf.discover_arm_hash(str(tmp_path), 43)
