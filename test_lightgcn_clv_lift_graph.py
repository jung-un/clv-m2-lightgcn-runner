import numpy as np
import pandas as pd

from lightgcn_clv_lift_graph import (
    CLVLiftGraphConfig,
    _build_clv_assignments,
    _compute_lift_edge_signal,
    configure_clv_lift_graph_run,
    preflight_summary,
    validate_config,
)


def test_lift_signal_is_clv_group_item_specific():
    train_pairs = pd.DataFrame(
        {
            "u_idx": [0, 1, 2, 2, 3],
            "i_idx": [0, 0, 0, 1, 1],
        }
    )
    groups = np.array([0, 0, 1, 1], dtype=np.int8)
    users, items, signal, _ = _compute_lift_edge_signal(
        train_pairs,
        groups,
        n_items=2,
        prior_strength=0.0,
        max_abs_log_lift=10.0,
    )
    actual = {(int(u), int(i)): float(s) for u, i, s in zip(users, items, signal)}

    assert np.isclose(actual[(0, 0)], np.log(5.0 / 3.0))
    assert np.isclose(actual[(2, 0)], np.log(5.0 / 9.0))
    assert np.isclose(actual[(2, 1)], np.log(5.0 / 3.0))


def test_shuffle_preserves_clv_values_inside_activity_deciles():
    rows = []
    for user in range(40):
        for basket in range(user + 1):
            basket_value = 1000.0 if user % 2 == 0 else 1.0
            rows.append((user, basket, basket_value))
    train = pd.DataFrame(rows, columns=["u_idx", "b_raw", "v"])
    assignments = _build_clv_assignments(train, n_users=40, seed=20260826)

    for decile in np.unique(assignments["n_decile"]):
        mask = assignments["n_decile"] == decile
        np.testing.assert_allclose(
            np.sort(assignments["q_clv"][mask]),
            np.sort(assignments["q_clv_shuffle"][mask]),
        )
    assert not np.array_equal(assignments["clv_group"], assignments["shuffle_group"])


def test_runner_is_locked_to_m3_graph_arms_and_historical_split(tmp_path):
    cfg = configure_clv_lift_graph_run(
        out_dir=str(tmp_path), baseline_result_dir=str(tmp_path / "baseline")
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [
        "m3_clv_lift_graph",
        "m3_clv_lift_graph_shuffle",
    ]
    assert summary["intervention_location"] == "graph edge weights only"
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False


def test_config_rejects_changed_epoch_or_final_test_split(tmp_path):
    base = dict(out_dir=str(tmp_path), baseline_result_dir=str(tmp_path / "base"))
    for cfg in (
        CLVLiftGraphConfig(epochs=99, **base),
        CLVLiftGraphConfig(time_cutoff=697, **base),
    ):
        try:
            validate_config(cfg)
            raise AssertionError("changed screen protocol must fail")
        except ValueError:
            pass
