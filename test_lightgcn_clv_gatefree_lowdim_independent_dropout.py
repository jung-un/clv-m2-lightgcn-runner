import pytest

import lightgcn_clv_gatefree_lowdim_independent_dropout as runner


def _config(tmp_path, **overrides):
    return runner.configure_independent_dropout_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
        **overrides,
    )


def test_preflight_fixes_independent_item_axes_and_four_training_states(tmp_path):
    cfg = _config(tmp_path)
    summary = runner.preflight_summary(cfg)
    m2 = summary["m2"]

    assert summary["trained_models"] == [
        "m2_gatefree_lowdim_independent_dropout"
    ]
    assert summary["reused_comparator"] == "m1_64"
    assert m2["architecture"] == "ID(64)|activity(4)|transaction-value(4)"
    assert m2["independent_item_axis_coordinates"] is True
    assert m2["fixed_per_axis_budget"] == pytest.approx(0.1)
    assert m2["axis_keep_probability"] == pytest.approx(0.5)
    assert m2["training_state_probabilities"] == {
        "ID_only": 0.25,
        "ID_plus_activity": 0.25,
        "ID_plus_transaction_value": 0.25,
        "full": 0.25,
    }
    assert m2["evaluation_score_formula"] == "S_ID + 0.1*S_N + 0.1*S_V"
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["loss"] == "one plain pairwise BPR; no added loss"
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_unplanned_budget_or_keep_probability_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="0.1"):
        _config(tmp_path, axis_budget=0.05)
    with pytest.raises(ValueError, match="0.5"):
        _config(tmp_path, axis_keep_probability=0.75)


def test_model_receives_independent_item_axes_and_dropout_without_item_features(
    tmp_path, monkeypatch
):
    cfg = _config(tmp_path)
    prepared = {
        "data": {"n_users": 3, "n_items": 4, "adj": None},
        "axes": {
            "activity": None,
            "value": None,
            "activity_valid": None,
            "value_valid": None,
            "q_n": None,
            "q_v": None,
        },
    }
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def to(self, _device):
            return self

        def parameters(self):
            return []

    monkeypatch.setattr(runner, "GateFreeLowDimNVLightGCN", FakeModel)
    runner._build_model(prepared, cfg)

    assert captured["axis_budget"] == pytest.approx(0.1)
    assert captured["training_axis_balance_delta"] == pytest.approx(0.0)
    assert captured["independent_item_axes"] is True
    assert captured["axis_keep_probability"] == pytest.approx(0.5)
    assert "item_profile" not in captured
