import numpy as np
import pandas as pd
import pytest

from clv_m4_interaction_weights import build_m4_interaction_weights
import lightgcn_clv_v3 as v3


def _toy_train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1],
            "i_idx": [0, 1, 0, 2],
            "b_raw": [10, 10, 20, 20],
            "v": [80.0, 20.0, 50.0, 50.0],
        }
    )


def test_clv_pair_weights_prioritize_high_contribution_inside_high_clv_user():
    result = build_m4_interaction_weights(
        _toy_train(), np.array([0.9, 0.1]), mode="clv_pair", beta=0.25
    )

    assert result.row_weights[0] > result.row_weights[1]
    assert result.row_weights[0] > result.row_weights[2]
    assert np.isclose(result.row_weights.mean(), 1.0)
    assert np.isfinite(result.row_weights).all()
    assert (result.row_weights > 0).all()


def test_pair_contribution_control_does_not_depend_on_clv_assignment():
    first = build_m4_interaction_weights(
        _toy_train(), np.array([0.9, 0.1]), mode="pair_contribution", beta=0.25
    )
    shuffled = build_m4_interaction_weights(
        _toy_train(), np.array([0.1, 0.9]), mode="pair_contribution", beta=0.25
    )

    np.testing.assert_allclose(first.row_weights, shuffled.row_weights)


def test_repeated_train_rows_for_same_pair_receive_same_weight():
    train = pd.concat([_toy_train(), _toy_train().iloc[[0]]], ignore_index=True)
    result = build_m4_interaction_weights(
        train, np.array([0.9, 0.1]), mode="clv_pair", beta=0.25
    )

    assert result.row_weights[0] == result.row_weights[-1]


@pytest.mark.parametrize("beta", [-0.1, np.nan, np.inf])
def test_interaction_weight_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        build_m4_interaction_weights(
            _toy_train(), np.array([0.9, 0.1]), mode="clv_pair", beta=beta
        )


def test_interaction_weight_diagnostics_include_required_audit_values():
    result = build_m4_interaction_weights(
        _toy_train(), np.array([0.9, 0.1]), mode="clv_pair", beta=0.25
    )

    assert set(
        [
            "weight_min",
            "weight_median",
            "weight_mean",
            "weight_max",
            "weight_std",
            "clv_weight_spearman",
        ]
    ).issubset(result.diagnostics)


@pytest.mark.parametrize("mode", ["pair_contribution", "clv_pair"])
def test_v3_loss_path_applies_interaction_weight_to_original_train_rows(mode):
    train = _toy_train().assign(up=lambda frame: frame["v"])
    cfg = {**v3.CFG, "LOSS_MODE": mode, "LOSS_LAMBDA": 0.25}

    weights = v3.build_loss_weights(
        train,
        train["u_idx"].to_numpy(np.int64),
        np.array([0.9, 0.1]),
        cfg,
    )

    assert len(weights) == len(train)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] > weights[1]


def test_v3_loss_weight_diagnostics_report_distribution_and_clv_assignment():
    diagnostics = v3.loss_weight_diagnostics(
        np.array([0.8, 1.2], dtype=np.float32),
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        "clv_pair",
    )

    assert diagnostics["mode"] == "clv_pair"
    assert diagnostics["min"] == pytest.approx(0.8)
    assert diagnostics["median"] == pytest.approx(1.0)
    assert diagnostics["mean"] == pytest.approx(1.0)
    assert diagnostics["max"] == pytest.approx(1.2)
    assert diagnostics["std"] == pytest.approx(0.2)
    assert diagnostics["clv_weight_spearman"] == pytest.approx(1.0)
