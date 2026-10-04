import numpy as np
import pandas as pd
import pytest
import torch

import clv_m5_m3_m4_split_nv_hm2y_m4_reference_300 as ref
import clv_m5_m3_m4_split_nv_hm2y_screen as m5
import lightgcn_clv_v3 as v3


METRICS = list(m5.ACCURACY) + list(m5.ECONOMIC)


def test_uniform_weights_must_rebuild_m1s_own_binary_adjacency():
    """A - reference only isolates the graph if the reference graph is M1's."""

    eu = np.array([0, 0, 1, 2], dtype=np.int64)
    ei = np.array([0, 1, 1, 0], dtype=np.int64)
    ones = np.ones(len(eu), dtype=np.float32)
    binary = v3.build_adj(eu, ei, ones, 3, 2)
    assert ref.same_adjacency(binary, v3.build_adj(eu, ei, ones, 3, 2))

    tilted = v3.build_adj(eu, ei, np.array([1, 1, 1, 4], np.float32), 3, 2)
    assert not ref.same_adjacency(binary, tilted)

    prepared = {"signals": {"edge_users": eu, "edge_items": ei},
                "data": {"n_users": 3, "n_items": 2, "adj": binary}}
    assert ref.binary_edge_weights(prepared).tolist() == [1.0] * 4

    prepared["data"]["adj"] = tilted
    with pytest.raises(RuntimeError, match="이진 인접행렬과 다릅니다"):
        ref.binary_edge_weights(prepared)


def _prepared(frozen_scale=1.0):
    row = {metric: 0.01 * frozen_scale for metric in METRICS}
    return {"m5_reference_curve": pd.DataFrame(
        [{"model_id": m5.M4_ID, "epoch": m5.JUDGE_EPOCH, **row}])}


def _payload(scale=1.0, epochs=(100,)):
    return {"model_id": ref.MODEL_ID, "stop_epoch": max(epochs),
            "curve": [{"epoch": e, "metrics": {m: 0.01 * scale for m in METRICS}}
                      for e in epochs]}


def test_float32_rounding_reproduces_but_a_real_gap_does_not():
    frame = ref.reproduction_gap(_payload(1 + 1e-6), _prepared())
    assert frame.attrs["reproduces_frozen_m4_at_100"] is True

    frame = ref.reproduction_gap(_payload(1.01), _prepared())   # 1% off
    assert frame.attrs["reproduces_frozen_m4_at_100"] is False
    assert frame.rel_gap.max() == pytest.approx(0.01, rel=1e-3)


def test_a_missing_epoch_100_evaluation_is_named():
    with pytest.raises(RuntimeError, match="epoch 100 평가가 없습니다"):
        ref.reproduction_gap(_payload(epochs=(300,)), _prepared())


def _comparison(arm_id, *, m1=1.00, reference=1.00, arm=1.05, epoch=300):
    rows = []
    for ref_id, scale in ((m5.M1_ID, m1), (ref.MODEL_ID, reference)):
        for metric in METRICS:
            base, value = 0.01 * scale, 0.01 * arm
            rows.append({"epoch": epoch, "model_id": arm_id, "reference": ref_id,
                         "metric": metric, "reference_value": base,
                         "candidate_value": value, "delta": value - base,
                         "ratio": value / base})
    return pd.DataFrame(rows)


def test_the_300_reading_separates_beating_m4_from_being_noninferior_to_it():
    arm = {"model_id": m5.ARMS["split_nv"], "stop_epoch": 300}
    reading = ref.arm_reading(_comparison(arm["model_id"], arm=1.05), arm, 300)
    assert reading["all_three_met"] is True
    assert reading["economic_at10_above_m4"] is True

    # 0.5% below M4 still clears the 99% non-inferiority line but does not beat it
    reading = ref.arm_reading(
        _comparison(arm["model_id"], arm=0.995), arm, 300)
    assert reading["economic_at10_noninferior_99pct_vs_m4"] is True
    assert reading["economic_at10_above_m4"] is False
    assert reading["both_economic_at10_above_m1"] is False


def test_an_arm_without_the_judged_epoch_is_not_silently_passed():
    arm = {"model_id": m5.ARMS["original"], "stop_epoch": 100}
    reading = ref.arm_reading(_comparison(arm["model_id"], epoch=100), arm, 300)
    assert reading == {"trained_to_epoch": 100, "evaluated": False}
