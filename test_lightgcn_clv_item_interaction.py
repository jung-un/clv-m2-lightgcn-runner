import numpy as np
import pandas as pd

from lightgcn_clv_item_interaction import (
    CLVItemInteractionConfig,
    _build_clv_coordinates,
    configure_clv_item_interaction_run,
    preflight_summary,
    validate_config,
)


def test_runner_is_locked_to_two_jointly_trained_arms_and_historical_split(tmp_path):
    cfg = configure_clv_item_interaction_run(
        out_dir=str(tmp_path), baseline_result_dir=str(tmp_path / "baseline")
    )
    summary = preflight_summary(cfg)
    assert summary["trained_models"] == [
        "m2_clv_item_interaction",
        "m2_clv_item_interaction_shuffle",
    ]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True


def test_config_rejects_changed_epoch_or_split(tmp_path):
    base = dict(out_dir=str(tmp_path), baseline_result_dir=str(tmp_path / "base"))
    try:
        validate_config(CLVItemInteractionConfig(epochs=99, **base))
        raise AssertionError("changed epochs must fail")
    except ValueError:
        pass
    try:
        validate_config(CLVItemInteractionConfig(time_cutoff=697, **base))
        raise AssertionError("changed split must fail")
    except ValueError:
        pass


def test_shuffle_preserves_clv_coordinates_within_n_decile():
    rows = []
    for user in range(20):
        for basket in range(user + 1):
            rows.append((user, basket, float((user + 1) * (basket + 1))))
    train = pd.DataFrame(rows, columns=["u_idx", "b_raw", "v"])
    actual, shuffled, diagnostics = _build_clv_coordinates(
        train, n_users=20, seed=20260826
    )

    assert np.isclose(actual.mean(), 0.0)
    assert np.isclose(shuffled.mean(), 0.0)
    for decile in np.unique(diagnostics["n_decile"]):
        mask = diagnostics["n_decile"] == decile
        np.testing.assert_allclose(np.sort(actual[mask]), np.sort(shuffled[mask]))
    assert diagnostics["shuffle_seed"] == 20260826
