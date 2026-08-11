import numpy as np
import pandas as pd
import torch

import lightgcn_clv_residual as residual


def _target_train() -> pd.DataFrame:
    rows = []
    for day in range(1, 51):
        rows.append(
            {
                "u_idx": 0,
                "i_idx": day % 4,
                "cat_idx": day % 2,
                "b_raw": f"u0-{day}",
                "t": day,
                "up": 2.0,
                "v": 2.0,
            }
        )
        if day % 2 == 0:
            rows.append(
                {
                    "u_idx": 1,
                    "i_idx": day % 3,
                    "cat_idx": day % 2,
                    "b_raw": f"u1-{day}",
                    "t": day,
                    "up": 5.0,
                    "v": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_anchor_targets_separate_future_transaction_count_and_value():
    anchors = residual.build_anchor_examples(
        _target_train(),
        n_users=2,
        is_date=False,
        input_days=10,
        target_days=4,
        anchor_offsets=(12, 8, 4),
    )
    anchor = anchors.anchors[-1]

    np.testing.assert_array_equal(anchor.transaction_target, [4.0, 2.0])
    np.testing.assert_allclose(anchor.mean_transaction_value_target, [2.0, 5.0])
    np.testing.assert_allclose(
        anchor.transaction_target * anchor.mean_transaction_value_target,
        anchor.amount_target,
    )


def test_clv_core_profile_contains_only_literature_axes_and_predictions():
    import clv_core_features as core

    train = _target_train()
    anchors = residual.build_anchor_examples(
        train,
        n_users=2,
        is_date=False,
        input_days=10,
        target_days=4,
        anchor_offsets=(12, 8, 4),
    )
    snapshot = residual.build_final_snapshot(train, 2, False, input_days=10)
    artifact = core.train_clv_core_encoder(
        anchors,
        snapshot,
        encoder_epochs=2,
        encoder_patience=1,
        encoder_batch_size=16,
        encoder_lr=1e-3,
        seed=42,
        device=torch.device("cpu"),
    )
    profile = core.compose_clv_core_profiles(
        artifact, snapshot, torch.device("cpu")
    )

    assert profile.values.shape == (2, 29)
    assert profile.feature_names[-3:] == (
        "pred_log_future_transactions",
        "pred_log_transaction_value",
        "pred_log_clv_proxy",
    )
    assert not {
        "premium_share",
        "category_entropy",
        "repeat_pair_share",
    }.intersection(profile.feature_names)
    np.testing.assert_allclose(
        artifact.ev_all, artifact.n_hat_all * artifact.v_hat_all, rtol=1e-5
    )
