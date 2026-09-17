import numpy as np
import pandas as pd

import lightgcn_clv_high_clv_n_quick_screen as screen


def test_fixed_quick_screen_contract_reuses_seed43_m1():
    cfg = screen.configure_quick_screen(
        out_dir="/tmp/out", m1_multiseed_result_dir="/tmp/m1"
    )
    assert cfg.seed == 43
    assert cfg.negative_count == 1
    assert cfg.epochs == 100
    assert cfg.reuse_exact_m1 is True
    assert screen.NEW_MODEL_IDS == (screen.M2_MODEL_ID, screen.M4_MODEL_ID)


def test_shrunk_item_buyer_context_avoids_single_buyer_identity():
    train = pd.DataFrame({"u_idx": [0, 0, 1], "i_idx": [0, 1, 1]})
    q_n = np.array([0.9, 0.1])
    built = screen.build_shrunk_buyer_context(
        train, q_n=q_n, n_items=2, prior_mass=10.0
    )
    assert 0.0 < built["item_mean"][0] < 0.9
    assert np.isclose(built["item_mean"][0], (0.9 + 10 * 0.5) / 11)
    assert built["single_buyer_item_share"] == 0.5


def test_positive_context_is_leave_one_user_out():
    train = pd.DataFrame({"u_idx": [0, 1, 2], "i_idx": [0, 0, 1]})
    q_n = np.array([0.2, 0.8, 1.0])
    built = screen.build_shrunk_buyer_context(
        train, q_n=q_n, n_items=2, prior_mass=2.0
    )
    values = screen.positive_leave_one_out_context(
        np.array([0, 1]), np.array([0, 0]), q_n, built, prior_mass=2.0
    )
    np.testing.assert_allclose(values, [(0.8 + 1.0) / 3.0, (0.2 + 1.0) / 3.0])


def test_pairwise_weights_are_high_clv_only_before_batch_normalization():
    users = np.array([0, 1, 0, 1])
    high_gate = np.array([1.0, 0.0])
    positive_fit = np.array([0.9, 0.9, 0.2, 0.2])
    negative_fit = np.array([0.1, 0.1, 0.8, 0.8])
    raw, normalized = screen.pairwise_n_weights(
        users, positive_fit, negative_fit, high_gate, lambda_=0.5
    )
    np.testing.assert_allclose(raw[[1, 3]], 1.0)
    np.testing.assert_allclose(normalized.mean(), 1.0)
    assert raw[0] > 1.0 and raw[2] == 1.0
