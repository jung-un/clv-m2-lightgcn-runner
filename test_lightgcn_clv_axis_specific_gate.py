from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from clv_dual_axis_model import apply_gate_shape
import lightgcn_clv_axis_specific_gate as axis_gate


def test_axis_specific_positive_gate_is_positive_ordered_and_mean_one():
    q = np.array([0.1, 0.9, 0.5, 0.0], np.float32)
    valid = np.array([True, True, True, False])

    gate = apply_gate_shape(q, "axis_positive", valid)

    assert gate[0] < gate[2] < gate[1]
    assert np.all(gate > 0)
    np.testing.assert_allclose(gate[valid].mean(), 1.0, rtol=1e-6)
    assert gate[~valid].item() == 1.0


def test_axis_specific_preset_is_dunnhumby_seed42_validation_only():
    cfg = axis_gate.configure_axis_specific_gate_dunnhumby_run()
    summary = axis_gate.preflight_summary(cfg)

    assert cfg.dataset == "dunnhumby"
    assert cfg.seed == 42
    assert cfg.gate_shape == "axis_positive"
    assert cfg.preference_preserving is True
    assert cfg.gamma_init == 0.1
    assert cfg.eval_test is False
    assert cfg.eval_holdout is False
    assert summary["models"] == ["m1", axis_gate.MODEL_ID]
    assert summary["graph_mode"] == "binary"
    assert summary["negative_sampling"] == "uniform"
    assert summary["m4_sample_weighting"] is False


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_axis_specific_preset_rejects_protected_splits(field):
    cfg = axis_gate.configure_axis_specific_gate_dunnhumby_run()
    with pytest.raises(ValueError):
        axis_gate.validate_axis_specific_gate_config(
            replace(cfg, **{field: True})
        )


def test_axis_specific_colab_is_pinned_and_runs_once_without_approval_gate():
    notebook = json.loads(
        Path("clv_m2_axis_specific_gate_dunnhumby_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    joined = "\n".join(sources)

    assert sum(
        source.strip() == "result_df = run_experiment(cfg)"
        for source in sources
    ) == 1
    assert "ACKNOWLEDGE_HIGH_COST" not in joined
    assert "4accfe76ace0c8bbd71adf936b66e8003160ec31" in joined
    assert "eval_test" not in joined.lower()
