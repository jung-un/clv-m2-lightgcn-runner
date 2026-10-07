import numpy as np

import clv_m5_m3_m4_user_centered_screen as screen


def test_only_preregistered_development_seeds_are_allowed():
    assert screen.configure(42).seeds == (42,)
    assert screen.configure(44).seeds == (44,)

    try:
        screen.configure(43)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("seed43 must remain outside this exact replication")


def test_customer_mass_is_preserved_and_invalid_rows_stay_plain():
    users = np.array([0, 0, 0, 1, 1, 2])
    q_c = np.array([1.0, 0.5, 0.9])
    term = np.array([0.1, 0.4, 0.9, 0.2, 0.8, 0.7])
    valid = np.array([True, True, False, True, True, True])

    weights, audit = screen.customer_centered_weights(users, q_c, term, valid, 3)

    assert np.allclose(np.bincount(users, weights=weights), np.bincount(users))
    assert weights[2] == 1.0
    assert weights[5] == 1.0  # one valid row centres to zero
    assert weights[0] < weights[1]
    assert weights[3] < weights[4]
    assert audit["invalid_rows_equal_one"]
    assert audit["customer_mass_ratio_cv"] < 1e-10


def test_strength_changes_priority_but_not_customer_mass():
    users = np.array([0, 0, 1, 1])
    q_c = np.array([0.8, 0.8])
    term = np.array([0.1, 0.9, 0.2, 0.8])
    valid = np.ones(4, dtype=bool)

    weak, _ = screen.customer_centered_weights(users, q_c, term, valid, 2, strength=.25)
    strong, _ = screen.customer_centered_weights(users, q_c, term, valid, 2, strength=.5)

    assert strong[1] / strong[0] > weak[1] / weak[0]
    assert np.allclose(np.bincount(users, weights=strong), [2.0, 2.0])


def test_comparison_adds_previous_arms_only_when_present():
    metrics = {metric: 1.0 for metric in (*screen.ACCURACY, *screen.ECONOMIC)}
    rows = []
    for epoch in (screen.DIAGNOSTIC_EPOCH, screen.FIXED_EPOCH):
        for model_id in (screen.m3.M1_MODEL_ID, screen.m3.ARM_VALUE_ACTIVITY, screen.MODEL_ID):
            rows.append(dict(model_id=model_id, seed=42, epoch=epoch, **metrics))
    basic = screen._comparison(screen.pd.DataFrame(rows), 42)
    assert set(basic.reference) == {screen.m3.M1_MODEL_ID, screen.m3.ARM_VALUE_ACTIVITY}

    for epoch in (screen.DIAGNOSTIC_EPOCH, screen.FIXED_EPOCH):
        for model_id in (screen.prior.ARM_A, screen.prior.ARM_B):
            rows.append(dict(model_id=model_id, seed=42, epoch=epoch, **metrics))
    expanded = screen._comparison(screen.pd.DataFrame(rows), 42)
    assert set(expanded.reference) == {
        screen.m3.M1_MODEL_ID, screen.m3.ARM_VALUE_ACTIVITY,
        screen.prior.ARM_A, screen.prior.ARM_B,
    }
