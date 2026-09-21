import json
from pathlib import Path

import pytest

import lightgcn_clv_candidate_nv_capacity_screen as screen


def test_capacity_config_locks_the_underfitting_experiment():
    cfg = screen.configure_capacity_screen(
        out_dir="/tmp/capacity", baseline_result_dir="/tmp/base"
    )
    summary = screen.preflight_summary(cfg)

    assert cfg.capacity_dimensions == (64, 128, 256)
    assert cfg.negative_count == 1
    assert cfg.n_layers == 2
    assert cfg.epochs == 100
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    with pytest.raises(ValueError, match="capacity_dimensions"):
        screen.configure_capacity_screen(
            out_dir="/tmp/capacity",
            baseline_result_dir="/tmp/base",
            capacity_dimensions=(64, 128),
        )


def test_nine_arms_match_m1_m2_m3_within_each_capacity():
    cfg = screen.configure_capacity_screen(
        out_dir="/tmp/capacity", baseline_result_dir="/tmp/base"
    )
    specs = screen.arm_specifications({"assignment": True}, cfg)

    assert len(specs) == 9
    assert [spec["model_id"] for spec in specs] == list(screen.MODEL_IDS)
    for dimension in (64, 128, 256):
        selected = [spec for spec in specs if spec["id_dim"] == dimension]
        assert [spec["m2"] for spec in selected] == [False, True, False]
        assert [spec["m3"] for spec in selected] == [False, False, True]
        assert all(spec["weighted"] is False for spec in selected)


def _metrics(value):
    return {metric: value for metric in screen.TOP10_METRICS}


def test_reading_requires_new_matched_capacity_advantage():
    rows = {}
    for dimension in screen.CAPACITY_DIMENSIONS:
        rows[screen.model_id("m1", dimension)] = _metrics(1.0)
        rows[screen.model_id("m2", dimension)] = _metrics(
            1.01 if dimension == 128 else 0.99
        )
        rows[screen.model_id("m3", dimension)] = _metrics(0.98)

    reading = screen.capacity_reading(rows)

    assert reading["capacity_underfitting_signal"] is True
    assert reading["newly_positive_larger_capacity"]["m2"] == [128]
    assert reading["newly_positive_larger_capacity"]["m3"] == []
    assert reading["clv_attribution_tested"] is False


def test_colab_runs_capacity_screen_without_test_or_holdout():
    path = Path("clv_candidate_nv_capacity_dunnhumby_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "run_capacity_screen" in source
    assert "(64, 128, 256)" in source
    assert "final_test_constructed" in source
    assert "holdout_constructed" in source
