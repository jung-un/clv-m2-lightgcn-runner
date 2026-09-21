import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_value_rerank_diagnostic as diag


def _cfg(**overrides):
    return diag.configure_rerank_diagnostic(out_dir="/tmp/value_rerank", **overrides)


class _Cache:
    """Minimal EvalCache stand-in: two users, one high-CLV and one low-CLV."""

    def __init__(self):
        self.users = np.array([0, 1])
        self.seg = np.array(["고CLV", "저CLV"])
        self.P_arr = np.array([4, 2])
        self.rev = {0: np.array([1.0, 2.0, 9.0, 4.0]), 1: np.array([1.0, 1.0])}


def _prepared(q_c, amounts):
    return {
        "cache": _Cache(),
        "q_c": np.asarray(q_c, dtype=np.float64),
        "item_expected_amount": np.asarray(amounts, dtype=np.float64),
    }


def _candidates(depth=50):
    # user 0 holds a cheap hit at rank 0 and an expensive hit at rank 30
    candidates = np.tile(np.arange(depth), (2, 1))
    values = np.zeros((2, depth))
    values[0, 0] = 1.0
    values[0, 30] = 9.0
    values[1, 1] = 1.0
    return candidates, values


def test_diagnostic_trains_nothing_and_keeps_the_unranked_control():
    summary = diag.preflight_summary(_cfg())

    assert summary["trains_anything"] is False
    assert summary["alphas"][0] == 0.0
    assert summary["split"] == "historical_development_days_684_690"


def test_config_rejects_a_different_slot_budget_or_a_missing_control():
    with pytest.raises(ValueError):
        _cfg(top_k=20)
    with pytest.raises(ValueError):
        _cfg(alphas=(0.2, 0.4))
    with pytest.raises(ValueError):
        _cfg(models=("m9_unknown",))


def test_headroom_splits_value_into_captured_reachable_and_out_of_reach():
    _, values = _candidates()
    frame = diag.headroom_table(values, _Cache(), _cfg())
    user0 = frame.iloc[0]

    assert user0["captured_value_10"] == 1.0
    assert user0["band_value_11_50"] == 9.0
    # the customer bought 16.0 in total, so 6.0 never appears in the top fifty
    assert user0["total_value"] == 16.0
    assert user0["beyond_value_50"] == 6.0
    # the ceiling takes the expensive hit into the top ten
    assert user0["oracle_value_10"] == 10.0
    assert user0["hits_10"] == 1 and user0["hits_50"] == 2


def test_alpha_zero_reproduces_the_model_ranking():
    candidates, values = _candidates()
    prepared = _prepared(q_c=[1.0, 1.0], amounts=np.linspace(1.0, 50.0, 50))

    frame = diag.rerank_table(candidates, values, prepared, _cfg())
    unchanged = frame[frame.alpha.eq(0.0)]

    assert unchanged.list_changed.sum() == 0
    assert float(unchanged.iloc[0]["value_10"]) == 1.0


def _amounts_with_one_expensive_item(depth=50, expensive=30):
    """A cheap catalog where the item the customer actually bought is the pricey one."""

    amounts = np.full(depth, 1.0)
    amounts[expensive] = 100.0
    return amounts


def test_the_push_is_gated_by_the_customers_own_clv():
    candidates, values = _candidates()
    # user 0 is a top-CLV customer, user 1 has no CLV weight at all
    prepared = _prepared(q_c=[1.0, 0.0], amounts=_amounts_with_one_expensive_item())

    frame = diag.rerank_table(candidates, values, prepared, _cfg(alphas=(0.0, 0.4)))
    pushed = frame[frame.alpha.eq(0.4)]

    assert float(pushed.iloc[0]["value_10"]) > 1.0
    assert bool(pushed.iloc[0]["list_changed"]) is True
    assert bool(pushed.iloc[1]["list_changed"]) is False


def test_a_price_only_push_can_buy_expensive_misses_instead_of_hits():
    """The rule ranks by expected amount, so a costly non-purchase can win a slot."""

    candidates, values = _candidates()
    # every deep candidate is expensive, but only rank 30 was actually bought
    prepared = _prepared(q_c=[1.0, 1.0], amounts=np.linspace(1.0, 50.0, 50))

    frame = diag.rerank_table(candidates, values, prepared, _cfg(alphas=(0.0, 0.4)))
    pushed = frame[frame.alpha.eq(0.4) & frame.user.eq(0)].iloc[0]

    assert bool(pushed["list_changed"]) is True
    assert float(pushed["value_10"]) == 1.0
    assert int(pushed["hits_10"]) == 1


def test_swept_value_never_beats_the_oracle_ceiling():
    candidates, values = _candidates()
    prepared = _prepared(q_c=[1.0, 1.0], amounts=_amounts_with_one_expensive_item())
    cfg = _cfg(alphas=(0.0, 0.1, 0.2, 0.4))

    ceiling = diag.headroom_table(values, _Cache(), cfg).iloc[0]["oracle_value_10"]
    swept = diag.rerank_table(candidates, values, prepared, cfg)

    assert swept[swept.user.eq(0)].value_10.max() <= ceiling


def test_reading_reports_the_ceiling_and_the_cost_without_selecting():
    cfg = _cfg(models=(recheck.M1_MODEL_ID,), alphas=(0.0, 0.4))
    head = pd.DataFrame(
        [{"model_id": recheck.M1_MODEL_ID, "segment": "고CLV",
          "captured_value_10_mean": 1.0, "band_value_11_50_mean": 1.0,
          "total_value_mean": 4.0, "oracle_value_10_mean": 2.0}]
    )
    rerank = pd.DataFrame(
        [{"model_id": recheck.M1_MODEL_ID, "segment": "고CLV", "alpha": 0.0,
          "value_10_mean": 1.0, "recall_10_mean": 0.10, "list_changed_mean": 0.0},
         {"model_id": recheck.M1_MODEL_ID, "segment": "고CLV", "alpha": 0.4,
          "value_10_mean": 1.5, "recall_10_mean": 0.08, "list_changed_mean": 0.9}]
    )

    reading = diag._reading(head, rerank, cfg)

    assert reading["high_clv_reachable_share"] == pytest.approx(0.25)
    assert reading["high_clv_ceiling_gain_share"] == pytest.approx(1.0)
    assert reading["best_swept_alpha_high_clv"] == 0.4
    assert reading["best_swept_value_gain_share"] == pytest.approx(0.5)
    assert reading["recall_cost_at_best_alpha"] == pytest.approx(-0.2)
    assert reading["alpha_selected"] is False
