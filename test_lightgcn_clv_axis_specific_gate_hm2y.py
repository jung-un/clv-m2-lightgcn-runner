from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import lightgcn_clv_axis_specific_gate_hm2y as hm2y


def test_hm2y_preset_keeps_selected_m2_and_protected_splits_closed():
    cfg = hm2y.configure_axis_specific_gate_hm2y_run()
    summary = hm2y.preflight_summary(cfg)

    assert cfg.dataset == "hm"
    assert cfg.window_days is None
    assert cfg.input_days == 365
    assert cfg.seed == 42
    assert cfg.gate_shape == "axis_positive"
    assert cfg.preference_preserving is True
    assert cfg.batch_size == 131_072
    assert cfg.eval_test is False
    assert cfg.eval_holdout is False
    assert summary["graph_mode"] == "binary"
    assert summary["negative_sampling"] == "uniform"
    assert summary["sample_weighting"] is False


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_hm2y_rejects_protected_splits(field):
    cfg = hm2y.configure_axis_specific_gate_hm2y_run()
    with pytest.raises(ValueError):
        hm2y.validate_hm2y_config(replace(cfg, **{field: True}))


def test_compact_checkpoint_excludes_static_buffers():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer("large_static_graph", torch.ones(100))

    state = hm2y._parameter_state(model)

    assert set(state) == {"weight"}


def test_hm2y_colab_runs_once_without_approval_gate():
    notebook = json.loads(
        Path("clv_m2_axis_specific_gate_hm2y_colab.ipynb").read_text(
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
    assert "77b47e48cd07fd71cc8170b9cb7e37c057508acc" in joined
    assert "TO_BE_PINNED_AFTER_REVIEW" not in joined
    assert "read_progress(cfg.out_dir)" in joined
