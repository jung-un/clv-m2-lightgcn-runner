import numpy as np
import pandas as pd
import torch

from clv_dataset_adaptive_orthogonal_nv_model import (
    DatasetAdaptiveOrthogonalNVLightGCN,
    dataset_parameter_initialization,
)
from clv_reliability_orthogonal_nv_model import build_features
import clv_m2_dataset_adaptive_orthogonal_nv_seed48 as screen


def _features(*, q_n=(0.1, 0.5, 0.9), q_v=(0.2, 0.6, 0.8)):
    frame = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2, 2],
            "i_idx": [0, 1, 1, 2, 2, 3],
            "up": [1.0, 2.0, 2.5, 4.0, 5.0, 8.0],
            "b_raw": [10, 11, 20, 21, 30, 31],
        }
    )
    return build_features(
        frame,
        n_users=3,
        n_items=5,
        q_n=np.asarray(q_n),
        q_v=np.asarray(q_v),
        valid=np.ones(3, bool),
    )


def _empty_adj(size):
    return torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long), torch.empty(0), (size, size)
    ).coalesce()


def test_train_only_initialization_is_positive_normalized_and_data_dependent():
    first = dataset_parameter_initialization(_features())
    second = dataset_parameter_initialization(
        _features(q_n=(0.49, 0.50, 0.51), q_v=(0.0, 0.5, 1.0))
    )

    assert np.isclose(sum(first["axis_weights"].values()), 3.0)
    assert all(value > 0 for value in first["axis_weights"].values())
    assert 0.03 <= first["rho"] <= 0.15
    assert first["axis_weights"] != second["axis_weights"]


def test_all_axis_and_strength_parameters_receive_plain_bpr_gradient():
    features = _features()
    torch.manual_seed(49)
    model = DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=3,
        n_items=5,
        features=features,
        adj=_empty_adj(8),
        n_layers=0,
    )
    users = torch.tensor([0, 1, 2])
    positives = torch.tensor([1, 2, 3])
    negatives = torch.tensor([4, 4, 0])

    loss, _ = model.bpr_loss(users, positives, negatives)
    loss.backward()

    assert model.axis_logits.grad is not None
    assert model.axis_logits.grad.norm() > 0
    assert model.rho_logit.grad is not None
    assert model.rho_logit.grad.abs() > 0
    diagnostics = model.representation_diagnostics()
    assert diagnostics["axis_weight_n"] > 0
    assert diagnostics["axis_weight_v"] > 0
    assert diagnostics["axis_weight_price"] > 0
    assert 0.03 <= diagnostics["adaptive_rho"] <= 0.15


def test_m2_rejects_bpr_row_weights():
    model = DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=3,
        n_items=5,
        features=_features(),
        adj=_empty_adj(8),
        n_layers=0,
    )
    users = torch.tensor([0, 1])
    positives = torch.tensor([1, 2])
    negatives = torch.tensor([4, 4])

    try:
        model.bpr_loss(users, positives, negatives, weights=torch.ones(2))
    except ValueError as error:
        assert "plain BPR" in str(error)
    else:
        raise AssertionError("M2 row weighting must be rejected")


def test_screen_trains_only_adaptive_m2_and_compares_two_references():
    assert [spec["model_id"] for spec in screen.specs()] == [screen.M2_ADAPTIVE]
    rows = []
    for model_id, factor in (
        (screen.M1, 1.0),
        (screen.M2_FIXED, 1.005),
        (screen.M2_ADAPTIVE, 1.02),
    ):
        rows.append(
            {
                "model_id": model_id,
                "seed": screen.SEED,
                "epoch": screen.FIXED_EPOCH,
                **{metric: factor for metric in (*screen.ACCURACY, *screen.ECONOMIC)},
            }
        )
    result = screen.reading(pd.DataFrame(rows))

    assert result["candidate_vs_m1"]
    assert result["all_at10_above_fixed_m2"]
    assert not result["significance_claim"]
    assert not result["clv_attribution_claim"]
