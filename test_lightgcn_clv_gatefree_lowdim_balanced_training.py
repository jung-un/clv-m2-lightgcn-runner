import pytest

import lightgcn_clv_gatefree_lowdim_balanced_training as runner


def _config(tmp_path, **overrides):
    return runner.configure_balanced_training_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
        **overrides,
    )


def test_preflight_fixes_complementary_training_and_unperturbed_evaluation(
    tmp_path,
):
    cfg = _config(tmp_path)
    summary = runner.preflight_summary(cfg)
    m2 = summary["m2"]

    assert summary["trained_models"] == [
        "m2_gatefree_lowdim_balanced_training"
    ]
    assert summary["reused_comparator"] == "m1_64"
    assert m2["fixed_per_axis_budget"] == pytest.approx(0.1)
    assert m2["training_axis_balance_delta"] == pytest.approx(0.3)
    assert m2["training_activity_coefficient_range"] == [0.07, 0.13]
    assert m2["training_transaction_value_coefficient_range"] == [0.07, 0.13]
    assert m2["training_total_axis_budget"] == pytest.approx(0.2)
    assert m2["same_epsilon_for_positive_and_negative_in_each_triplet"] is True
    assert m2["evaluation_score_formula"] == "S_ID + 0.1*S_N + 0.1*S_V"
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_unplanned_budget_or_delta_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="0.1"):
        _config(tmp_path, axis_budget=0.05)
    with pytest.raises(ValueError, match="0.3"):
        _config(tmp_path, training_axis_balance_delta=0.2)


def test_model_receives_training_delta_without_item_features(tmp_path, monkeypatch):
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
    assert captured["training_axis_balance_delta"] == pytest.approx(0.3)
    assert "item_profile" not in captured
