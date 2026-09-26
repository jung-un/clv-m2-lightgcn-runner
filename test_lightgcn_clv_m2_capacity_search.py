import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m2_capacity_search as search


def _cfg(**overrides):
    return search.configure_capacity_search(out_dir="/tmp/capacity_search", **overrides)


def test_the_protocol_epoch_stays_on_the_evaluation_grid():
    summary = search.preflight_summary(_cfg())

    assert summary["protocol_epoch"] == 100
    assert 100 in summary["evaluated_at_epochs"]
    assert summary["evaluated_at_epochs"][-1] == summary["epochs"] == 300
    assert summary["loss"]["negative_count"] == 1


def test_config_rejects_a_budget_that_cannot_answer_the_question():
    with pytest.raises(ValueError):
        _cfg(epochs=150)          # 기존 100 epoch의 두 배 미만
    with pytest.raises(ValueError):
        _cfg(eval_every=40)       # 100 지점이 격자에서 빠진다
    with pytest.raises(ValueError):
        _cfg(negative_count=5)


def test_m1_is_retrained_once_per_shared_setting_not_once_per_condition():
    specs = search.arm_specifications(_cfg())
    m1 = [s for s in specs if s["model_id"] == search.M1_MODEL_ID]
    m2 = [s for s in specs if s["model_id"] == search.M2_MODEL_ID]

    # five conditions, but three of them share the 64-dim / L2 1e-3 M1
    assert len(m2) == len(search.CONDITIONS) == 5
    assert {search.shared_baseline(s) for s in m1} == {
        "dim64_l20.001", "dim128_l20.001", "dim64_l20.0001"
    }
    assert len(m1) == 3
    # every M2 condition has an M1 trained under the same shared setting
    assert {search.shared_baseline(s) for s in m2} <= {search.shared_baseline(s) for s in m1}


def test_each_condition_changes_exactly_one_knob_from_the_baseline():
    baseline = search.CONDITIONS[0]
    for condition in search.CONDITIONS[1:]:
        changed = sum(
            getattr(condition, field) != getattr(baseline, field)
            for field in ("id_dim", "pref_reg", "rho", "axis_dim")
        )
        assert changed == 1, condition.name
        assert condition.hypothesis != baseline.hypothesis


def test_intervention_strength_is_not_reported_as_an_underfitting_test():
    summary = search.preflight_summary(_cfg())

    assert summary["not_an_underfitting_test"] == ["strong_signal"]
    assert "intervention_strength" in summary["hypotheses"]


def test_every_condition_reaches_the_comparison_table():
    """A condition that only changes an M2-side knob shares M1 with the baseline."""

    cfg = _cfg()
    rows = []
    for spec in search.arm_specifications(cfg):
        rows.append(
            {**{k: spec[k] for k in ("condition", "hypothesis", "shared_key",
                                     "model_id", "id_dim", "axis_dim", "pref_reg", "rho")},
             "seed": 42, "epoch": 100, "loss": 0.1, "p_correct": 0.9,
             "recall@10": 0.010 if spec["model_id"] == search.M1_MODEL_ID else 0.011,
             "ndcg@10": 0.02}
        )
    gap = search.gap_table(pd.DataFrame(rows))

    assert set(gap.condition) == {c.name for c in search.CONDITIONS}
    assert set(gap[gap.condition.eq("strong_signal")].shared_key) == {"dim64_l20.001"}


def test_gap_table_refuses_to_silently_drop_an_unpaired_condition():
    cfg = _cfg()
    rows = [
        {**{k: spec[k] for k in ("condition", "hypothesis", "shared_key",
                                 "model_id", "id_dim", "axis_dim", "pref_reg", "rho")},
         "seed": 42, "epoch": 100, "loss": 0.1, "p_correct": 0.9, "recall@10": 0.01}
        for spec in search.arm_specifications(cfg)
        if spec["model_id"] != search.M1_MODEL_ID          # M1 결과가 통째로 빠진 상황
    ]
    with pytest.raises(KeyError, match="M1"):
        search.gap_table(pd.DataFrame(rows))


def _curve(m1_recall, m2_recall, condition="baseline", epochs=(100, 300),
           shared_key="dim64_l20.001"):
    rows = []
    for model_id, values in ((search.M1_MODEL_ID, m1_recall), (search.M2_MODEL_ID, m2_recall)):
        for epoch, value in zip(epochs, values):
            rows.append(
                {"condition": condition, "hypothesis": "training_budget",
                 "shared_key": shared_key, "model_id": model_id, "id_dim": 64,
                 "axis_dim": 4, "pref_reg": 1e-3, "rho": 0.0, "seed": 42, "epoch": epoch,
                 "loss": 0.1, "p_correct": 0.96,
                 "recall@10": value, "ndcg@10": value * 1.2}
            )
    return pd.DataFrame(rows)


def test_gap_table_pairs_m2_against_its_own_condition_baseline():
    curve = _curve([0.010, 0.012], [0.009, 0.013])
    gap = search.gap_table(curve)

    assert list(gap.epoch) == [100, 300]
    assert gap.iloc[0]["recall@10"] == pytest.approx(-0.001)
    assert gap.iloc[1]["recall@10"] == pytest.approx(0.001)
    assert gap.iloc[1]["recall@10_ratio"] == pytest.approx(0.013 / 0.012)


def test_reading_detects_training_that_had_not_finished_at_the_protocol_epoch():
    curve = _curve([0.010, 0.012], [0.009, 0.013])
    reading = search.search_reading(curve, search.gap_table(curve), _cfg())

    assert reading["m1_peak_epoch"] == 300
    assert reading["m1_still_improving_after_protocol_epoch"] is True
    assert reading["gap_closes_with_training"] is True
    assert reading["condition_selected"] is False
    assert reading["seeds_used"] == [42]


def test_reading_detects_a_peak_before_the_protocol_epoch():
    # development accuracy peaked early and the gap widened with more training
    curve = _curve([0.012, 0.010], [0.009, 0.006])
    reading = search.search_reading(curve, search.gap_table(curve), _cfg())

    assert reading["m1_peak_epoch"] == 100
    assert reading["m1_still_improving_after_protocol_epoch"] is False
    assert reading["gap_closes_with_training"] is False


def test_shortlist_names_conditions_that_beat_the_baseline_gap():
    baseline = _curve([0.010, 0.012], [0.009, 0.0095])
    wide = _curve([0.010, 0.012], [0.009, 0.014], condition="wide",
                  shared_key="dim128_l20.001")
    curve = pd.concat([baseline, wide], ignore_index=True)
    reading = search.search_reading(curve, search.gap_table(curve), _cfg())

    assert reading["conditions_closing_the_gap"] == ["wide"]
    assert reading["condition_selected"] is False


def test_running_one_condition_first_keeps_the_others_reusable():
    full = _cfg()
    staged = _cfg(conditions=("baseline",))

    assert search._config_hash(full, "input-hash") == search._config_hash(staged, "input-hash")
    assert search._config_hash(full, "other-input") != search._config_hash(full, "input-hash")
    assert [s["condition"] for s in search.arm_specifications(staged)] == ["baseline", "baseline"]

    light_l2 = _cfg(conditions=("light_l2",))
    assert [s["condition"] for s in search.arm_specifications(light_l2)] == [
        "light_l2", "light_l2"
    ]
    with pytest.raises(ValueError):
        _cfg(conditions=("baseline", "unknown"))
    with pytest.raises(ValueError):
        _cfg(conditions=("wide", "light_l2"))


def test_single_nonbaseline_condition_uses_itself_as_learning_curve_reference():
    light_l2 = _cfg(conditions=("light_l2",))
    curve = _curve(
        [0.010, 0.012], [0.009, 0.013],
        condition="light_l2", shared_key="dim64_l20.0001"
    )
    reading = search.search_reading(curve, search.gap_table(curve), light_l2)

    assert reading["reference_condition"] == "light_l2"


def test_curve_attrs_stay_displayable():
    """A DataFrame in attrs breaks display() for any frame wider than 20 columns."""

    curve = _curve([0.010, 0.012], [0.009, 0.013])
    gap = search.gap_table(curve)
    curve.attrs.update(
        reading={"ok": True}, result_paths={"json": "/tmp/x.json"},
        gap_records=gap.to_dict("records"),
    )

    assert not any(isinstance(value, pd.DataFrame) for value in curve.attrs.values())
    assert list(search.gap_frame(curve).epoch) == list(gap.epoch)

    wide = pd.concat([curve] + [curve[["recall@10"]].rename(columns={"recall@10": f"c{i}"})
                                for i in range(30)], axis=1)
    wide.attrs = dict(curve.attrs)
    wide.to_string()          # pandas truncates here; a DataFrame in attrs would raise


def test_the_log_label_works_for_both_runners_that_share_this_loop():
    """The M3 graph runner reuses _train_curve and its specs carry `arm`."""

    m2 = next(s for s in search.arm_specifications(_cfg())
              if s["model_id"] == search.M2_MODEL_ID)
    assert search.run_label(m2, 42) == f"baseline/{search.M2_MODEL_ID} s42"

    m3_spec = {"model_id": "m3_centered_value_graph_bpr_k1", "arm": "value_only"}
    assert search.run_label(m3_spec, 43) == "value_only/m3_centered_value_graph_bpr_k1 s43"

    with pytest.raises(KeyError):
        search.run_label({"model_id": "x"}, 42)
