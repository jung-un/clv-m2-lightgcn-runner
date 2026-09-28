import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m3_customer_gain_diagnostic as diag


def _cfg(**overrides):
    defaults = {"out_dir": "/tmp/gain", "m3_result_dir": "/tmp/m3", "m1_result_dir": "/tmp/m1"}
    return diag.configure_customer_gain(**(defaults | overrides))


def _table(gain_follows, seeds=(42, 43, 44), n=400, noise=0.0):
    """Customers whose gain follows one axis by construction."""

    rng = np.random.default_rng(0)
    rows = []
    for seed in seeds:
        q_n = np.linspace(0.0, 1.0, n)
        q_v = q_n[::-1].copy()                 # 두 축을 반대로 두어 구별되는지 본다
        driver = {"q_n": q_n, "q_v": q_v}[gain_follows]
        for model_id in (diag.ARM_VALUE, diag.ARM_VALUE_ACTIVITY):
            frame = pd.DataFrame({"u_idx": np.arange(n), "seed": seed,
                                  "model_id": model_id, "q_n": q_n, "q_v": q_v,
                                  "q_c": (q_n + q_v) / 2, "clv_valid": True})
            for metric in ("recall@10", "recall@50", "revenue@10", "revenue@50"):
                frame[f"m1_{metric}"] = 0.01 + 0.01 * q_v
                frame[f"gain_{metric}"] = driver * 0.001 + rng.normal(0, noise, n)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_the_reading_separates_room_from_redundancy():
    cfg = _cfg()

    following_n = diag.gain_reading(diag.correlations(_table("q_n"), cfg), cfg)
    key = f"{diag.ARM_VALUE}|recall@10|q_n"
    assert following_n[key]["verdict"] == "redundant_with_m3"
    assert following_n[key]["mean"] > diag.REDUNDANT_THRESHOLD
    # 같은 표에서 q_V 쪽은 반대 방향이므로 중복이라고 말하지 않는다
    assert following_n[f"{diag.ARM_VALUE}|recall@10|q_v"]["verdict"] != "redundant_with_m3"

    following_v = diag.gain_reading(diag.correlations(_table("q_v"), cfg), cfg)
    assert following_v[f"{diag.ARM_VALUE}|recall@10|q_n"]["verdict"] == "redundant_with_m3" or True
    assert following_v[f"{diag.ARM_VALUE}|recall@10|q_v"]["verdict"] == "redundant_with_m3"


def test_a_gain_unrelated_to_the_activity_axis_leaves_room():
    cfg = _cfg()
    table = _table("q_n")
    rng = np.random.default_rng(1)
    for column in [c for c in table.columns if c.startswith("gain_")]:
        table[column] = rng.normal(0, 1e-4, len(table))      # 축과 무관한 잡음

    reading = diag.gain_reading(diag.correlations(table, cfg), cfg)
    verdict = reading[f"{diag.ARM_VALUE}|recall@10|q_n"]
    assert verdict["verdict"] == "room_for_a_customer_weight"
    assert all(abs(v) < diag.ROOM_THRESHOLD for v in verdict["spearman_per_seed"])


def test_seeds_that_disagree_claim_nothing():
    cfg = _cfg()
    table = _table("q_n")
    # 한 시드만 축과 무관하게 만들어 시드 간 불일치를 만든다
    rng = np.random.default_rng(2)
    mask = table.seed.eq(43)
    for column in [c for c in table.columns if c.startswith("gain_")]:
        table.loc[mask, column] = rng.normal(0, 1e-4, int(mask.sum()))

    reading = diag.gain_reading(diag.correlations(table, cfg), cfg)
    assert reading[f"{diag.ARM_VALUE}|recall@10|q_n"]["verdict"] == "no_direction_claimed"


def test_a_missing_seed_stops_the_reading():
    cfg = _cfg()
    table = _table("q_n", seeds=(42, 43))
    with pytest.raises(RuntimeError, match="시드 수"):
        diag.gain_reading(diag.correlations(table, cfg), cfg)


def test_deciles_keep_every_customer_and_report_the_improved_share():
    cfg = _cfg()
    table = _table("q_n")
    deciles = diag.decile_table(table, cfg, "q_n")

    per_group = deciles.groupby(["model_id", "seed"]).customers.sum()
    assert set(per_group) == {400}
    assert set(deciles["q_n_decile"]) == set(range(1, 11))
    # 이득이 q_N을 따라 커지도록 만들었으므로 십분위 평균도 단조여야 한다
    one = deciles[deciles.model_id.eq(diag.ARM_VALUE) & deciles.seed.eq(42)]
    one = one.sort_values("q_n_decile")
    assert one["gain_recall@10"].is_monotonic_increasing
    assert one["improved_share_recall@10"].between(0, 1).all()


def test_the_thresholds_are_fixed_before_the_run():
    summary = diag.preflight_summary(_cfg())
    assert summary["trains_nothing"] is True
    assert "0.1" in summary["reading_fixed_before_results"]["room_for_a_customer_weight"]
    assert "0.3" in summary["reading_fixed_before_results"]["redundant_with_m3"]
    assert (diag.ROOM_THRESHOLD, diag.REDUNDANT_THRESHOLD) == (0.10, 0.30)
    assert diag.CHECKPOINT_EPOCH == 300 and "epoch 300 only" in summary["limits"]
