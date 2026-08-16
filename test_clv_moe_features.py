import numpy as np
import pandas as pd
import torch

import lightgcn_clv_residual as residual


def _encoder_case():
    model = residual.FutureValueEncoder(input_dim=32)
    transform = residual.FeatureTransform(
        mean=np.zeros(16, np.float32), std=np.ones(16, np.float32)
    )
    artifact = residual.EncoderArtifact(
        model=model,
        transform=transform,
        best_epoch=1,
        diagnostics={},
        h_all=np.zeros((3, 16), np.float32),
        ev_all=np.zeros(3, np.float32),
    )
    snapshot = residual.AnchorExamples(
        offset_days=0,
        observation_start=0,
        observation_end=365,
        target_start=366,
        target_end=365,
        user_ids=np.array([0, 1], np.int64),
        numeric=np.arange(32, dtype=np.float32).reshape(2, 16),
        valid=np.ones((2, 16), bool),
        purchase_target=np.zeros(2, np.float32),
        amount_target=np.zeros(2, np.float32),
    )
    return artifact, snapshot


def _tiny_train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 1, 2],
            "i_idx": [0, 0, 0, 1, 2, 3],
            "cat_idx": [10, 10, 10, 20, 20, 30],
            "up": [1.0, 1.0, 1.2, 3.0, 4.0, 8.0],
            "v": [1.0, 1.0, 1.2, 3.0, 4.0, 8.0],
        }
    )


def test_user_profile_contains_behavior_masks_hidden_and_predictions():
    import clv_moe_features as features

    artifact, snapshot = _encoder_case()
    out = features.compose_user_profiles(artifact, snapshot, torch.device("cpu"))
    assert out.values.shape == (3, 51)
    np.testing.assert_array_equal(out.valid_user, [True, True, False])
    assert len(out.feature_names) == 51
    assert np.isfinite(out.values).all()
    np.testing.assert_array_equal(out.values[2], np.zeros(51, np.float32))


def test_item_profiles_are_train_only_finite_and_category_encoded():
    import clv_moe_features as features

    out = features.build_item_profiles(_tiny_train(), n_items=5)
    assert out.numeric.shape == (5, 6)
    assert out.category_ids.shape == (5,)
    assert out.n_categories == 4  # unknown plus three observed categories
    assert np.isfinite(out.numeric).all()
    np.testing.assert_array_equal(out.valid_item, [True, True, True, True, False])
    assert out.category_ids[4] == 0


def test_item_profile_values_depend_only_on_supplied_train_rows():
    import clv_moe_features as features

    train = _tiny_train()
    validation = train.copy()
    validation["up"] = 9999.0
    before = features.build_item_profiles(train, 5)
    after = features.build_item_profiles(train.copy(), 5)
    np.testing.assert_array_equal(before.numeric, after.numeric)
    assert not np.array_equal(
        before.numeric, features.build_item_profiles(validation, 5).numeric
    )


def test_item_profiles_reject_out_of_range_item_indices():
    import clv_moe_features as features

    train = _tiny_train()
    train.loc[0, "i_idx"] = 5
    try:
        features.build_item_profiles(train, n_items=5)
    except ValueError as exc:
        assert "n_items" in str(exc)
    else:
        raise AssertionError("out-of-range item index must be rejected")
