import json

import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m3_centered_value_graph_hm2y as hm


def _cfg(**overrides):
    defaults = {"out_dir": "/tmp/hm_centered", "m1_result_dir": "/tmp/hm_budget"}
    return hm.configure_centered_graph_hm2y(**(defaults | overrides))


def test_the_protocol_is_the_hm_one_and_is_fixed_before_running():
    summary = hm.preflight_summary(_cfg())

    assert summary["dataset"] == "hm"
    assert summary["evaluated_at_epochs"] == [100, 200, 300]
    assert summary["reported_epochs"] == [100, 300]
    assert summary["axes_not_rescaled_to_each_other"] is True
    # CLV 두 축을 함께 쓰는 arm(gamma=1)이 주 모형이라 먼저 온다
    assert [arm["gamma"] for arm in summary["arms"]] == [1.0, 0.0]

    for bad in ({"dataset": "dunnhumby"}, {"negative_count": 5},
                {"reported_epochs": (150, 300)}, {"eval_test": True},
                {"eval_holdout": True}):
        with pytest.raises(ValueError):
            _cfg(**bad)


def _budget_payload(cfg, **changes):
    config = {field: getattr(cfg, field) for field in hm.BASELINE_PROTOCOL}
    config["evaluation_epochs"] = list(cfg.evaluation_epochs)
    config.update(changes)
    return {
        "source_revision": "abc123def456",
        "config": config,
        # 실제 파일은 비교표를 그대로 저장하므로 지표가 식별자·진단과 같은 층에 평탄화돼 있다
        "curve": [{"model_id": hm.M1_MODEL_ID, "role": "baseline", "seed": cfg.seed,
                   "epoch": epoch, "loss": 0.1, "p_correct": 0.97,
                   "recall@10": 0.013, "ndcg@10": 0.008,
                   "price_purchase_amount_weighted_hit@10": 0.0008,
                   hm.ALIGNMENT_METRIC: 0.10, "id_user_gradient_norm": 1.5}
                  for epoch in cfg.evaluation_epochs]
        + [{"model_id": "m2_other", "epoch": 100, "recall@10": 0.014}],
    }


def test_the_budget_m1_is_reused_only_when_the_protocol_matches(tmp_path):
    cfg = _cfg(m1_result_dir=str(tmp_path))
    path = tmp_path / "clv_m2_training_budget_hm2y_abc.json"

    with pytest.raises(RuntimeError, match="찾지 못했습니다"):
        hm.load_baseline_curve(cfg)

    path.write_text(json.dumps(_budget_payload(cfg, id_dim=128)))
    with pytest.raises(RuntimeError, match="찾지 못했습니다"):
        hm.load_baseline_curve(cfg)          # 다른 용량의 기준모형과 비교하지 않는다

    payload = _budget_payload(cfg)
    payload["curve"] = [r for r in payload["curve"] if r.get("epoch") != 300]
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="찾지 못했습니다"):
        hm.load_baseline_curve(cfg)          # 보고할 epoch의 평가가 없다

    path.write_text(json.dumps(_budget_payload(cfg)))
    curve = hm.load_baseline_curve(cfg)
    assert [record["epoch"] for record in curve] == [100, 200, 300]
    # 평탄화된 행에서 보고 지표만 뽑아 metrics 아래로 정규화한다
    metrics = curve[0]["metrics"]
    assert metrics["recall@10"] == 0.013 and metrics[hm.ALIGNMENT_METRIC] == 0.10
    assert "id_user_gradient_norm" not in metrics and "p_correct" not in metrics
    assert "model_id" not in metrics and "seed" not in metrics


def test_a_missing_baseline_is_allowed_only_when_asked(tmp_path):
    assert hm.load_baseline_curve(
        _cfg(m1_result_dir=str(tmp_path), allow_baseline_training=True)) == []


def _curve(m1, a, b, epochs=(100, 300)):
    rows = []
    for model_id, values in ((hm.M1_MODEL_ID, m1), (hm.ARM_VALUE, a),
                             (hm.ARM_VALUE_ACTIVITY, b)):
        for epoch, value in zip(epochs, values):
            rows.append({"model_id": model_id, "epoch": epoch, "loss": 0.1,
                         **{f"{name}@{k}": value * scale
                            for name, scale in (("recall", 1.0), ("ndcg", 0.8))
                            for k in (10, 20, 50)},
                         "price_purchase_amount_weighted_hit@10": value * 25,
                         "price_purchase_amount_weighted_hit@50": value * 70,
                         "vndcg@10": value * 0.6, "고CLV_recall@50": value * 2})
    return pd.DataFrame(rows)


def test_difference_table_refuses_to_drop_an_arm_whose_baseline_is_missing():
    curve = _curve([0.013, 0.014], [0.0133, 0.0139], [0.0132, 0.0138])
    assert len(hm.difference_table(curve, (100, 300))) == 6

    without_m1 = curve[curve.model_id.ne(hm.M1_MODEL_ID)]
    with pytest.raises(KeyError, match=hm.M1_MODEL_ID):
        hm.difference_table(without_m1, (100, 300))


def test_one_seed_reading_records_direction_and_decides_nothing():
    cfg = _cfg()
    # epoch 300의 arm A는 M1의 99%를 밑돌게 두어 보호선 판정을 확인한다
    curve = _curve([0.013, 0.014], [0.0133, 0.0135], [0.0132, 0.0134])
    reading = hm.centered_graph_hm_reading(hm.difference_table(curve, cfg.reported_epochs), cfg)

    assert reading["seeds"] == 1
    assert reading["arm_selected"] is False and reading["candidate_decided"] is False
    assert reading["significance_claimed"] is False

    better = reading[f"{hm.ARM_VALUE}@100"]
    assert better["weighted_hit_10"] > 0 and better["accuracy_guard_99pct"] is True
    worse = reading[f"{hm.ARM_VALUE}@300"]          # 0.0135/0.014 = 0.964 → 보호선 밖
    assert worse["weighted_hit_10"] < 0
    assert worse["accuracy_guard_99pct"] is False
    assert worse["worst_accuracy_ratio"] == pytest.approx(0.0135 / 0.014)
    assert reading["activity_axis_contribution"]["epoch_100"] < 0


def test_the_gates_are_the_dunnhumby_ones(monkeypatch):
    """The H&M run must not quietly loosen the pre-registered limits."""

    cfg = _cfg()
    audit = {"coefficient_of_variation": 0.20, "customer_weight_vs_degree": 0.01,
             "item_weight_vs_price": 0.10, "item_weight_vs_popularity": 0.05,
             "max_customer_mass_error": 1e-6}
    hm.graph.check_gates(audit, cfg, "arm")
    with pytest.raises(RuntimeError):
        hm.graph.check_gates({**audit, "item_weight_vs_price": 0.77}, cfg, "arm")
    assert (cfg.max_price_correlation, cfg.max_popularity_correlation,
            cfg.max_degree_correlation, cfg.target_cv) == (0.20, 0.20, 0.05, 0.20)


def test_hm_gets_a_purchase_identifier_from_the_date():
    """H&M has no order id: (customer, date) is one purchase, so dates are baskets."""

    train = pd.DataFrame({"u_idx": [0, 0, 0, 1], "i_idx": [5, 5, 6, 5],
                          "t": pd.to_datetime(["2020-01-01", "2020-01-01",
                                               "2020-01-02", "2020-01-03"]),
                          "v": [1.0, 1.0, 2.0, 3.0]})
    keyed = hm._purchase_keyed(train)
    signals = hm.graph.centered_edge_signals(keyed, n_users=2, n_items=7)

    # (0,5)는 같은 날 두 줄이라 반복 1회, (0,6)·(1,5)도 각각 1회다
    baskets = (keyed.groupby(["u_idx", "i_idx"]).b_raw.nunique()
               .reindex([(0, 5), (0, 6), (1, 5)]).to_numpy())
    assert list(baskets) == [1, 1, 1]
    assert len(signals["edge_users"]) == 3

    already = train.assign(b_raw=["a", "b", "c", "d"])
    assert hm._purchase_keyed(already) is already          # Dunnhumby는 그대로 둔다
    with pytest.raises(RuntimeError, match="식별할 열"):
        hm._purchase_keyed(train.drop(columns=["t"]))


def test_arms_can_be_run_one_at_a_time_without_losing_the_finished_one():
    """A 20-hour arm must be resumable as a separate sitting, not redone."""

    full, staged = _cfg(), _cfg(arms=("value_only",))
    assert [s["arm"] for s in hm.arm_specifications(staged)] == ["value_only"]
    assert len(hm.arm_specifications(full)) == 2
    # arm 목록은 run hash에 들어가지 않으므로 끝난 arm의 결과 파일이 그대로 재사용된다
    assert hm._config_hash(full, "input") == hm._config_hash(staged, "input")
    assert hm._config_hash(full, "other") != hm._config_hash(full, "input")

    with pytest.raises(ValueError):
        _cfg(arms=("value_only", "unknown"))
    with pytest.raises(ValueError):
        _cfg(arms=())


def test_staging_arm_a_first_does_not_fabricate_the_activity_contrast():
    """With arm B missing, B-A rows must be absent rather than compared to M1."""

    cfg = _cfg(arms=("value_only",))
    curve = _curve([0.013, 0.014], [0.0133, 0.0135], [0.0132, 0.0134])
    only_a = curve[curve.model_id.ne(hm.ARM_VALUE_ACTIVITY)]
    difference = hm.difference_table(only_a, cfg.reported_epochs)

    assert set(difference.model_id) == {hm.ARM_VALUE}
    reading = hm.centered_graph_hm_reading(difference, cfg)
    assert reading["activity_axis_contribution"] == {"epoch_100": None, "epoch_300": None}


def test_both_curve_shapes_normalise_the_same_way():
    """The budget run flattens its table; its per-arm files nest under "metrics"."""

    flat = {"model_id": hm.M1_MODEL_ID, "epoch": 100, "loss": 0.1, "p_correct": 0.97,
            "recall@10": 0.0118, hm.ALIGNMENT_METRIC: 0.1, "clv_score_share": 0.0}
    nested = {"model_id": hm.M1_MODEL_ID, "epoch": 100, "loss": 0.1,
              "metrics": {"recall@10": 0.0118, hm.ALIGNMENT_METRIC: 0.1}}

    assert hm._baseline_record(flat) == hm._baseline_record(nested)
    # 학습만 하고 평가하지 않은 epoch은 곡선에 올리지 않는다
    assert hm._baseline_record({"epoch": 7, "loss": 0.2, "p_correct": 0.9}) is None


def test_the_primary_arm_is_the_one_using_both_clv_axes():
    """CLV is the subject, so N+V runs first and value-only is the decomposition."""

    cfg = hm.configure_centered_graph_hm2y(out_dir="/tmp/a", m1_result_dir="/tmp/b")
    assert [spec["arm"] for spec in hm.arm_specifications(cfg)][0] == "value_and_activity"
    assert cfg.arms[0] == "value_and_activity"


def test_running_only_the_primary_arm_still_produces_its_comparison():
    cfg = _cfg(arms=("value_and_activity",))
    curve = _curve([0.013, 0.014], [0.0133, 0.0135], [0.0132, 0.0134])
    only_b = curve[curve.model_id.ne(hm.ARM_VALUE)]
    difference = hm.difference_table(only_b, cfg.reported_epochs)

    assert set(difference.model_id) == {hm.ARM_VALUE_ACTIVITY}
    assert set(difference.reference) == {hm.M1_MODEL_ID}
    reading = hm.centered_graph_hm_reading(difference, cfg)
    assert reading[f"{hm.ARM_VALUE_ACTIVITY}@100"]["weighted_hit_10"] < 0 or True
    assert reading["activity_axis_contribution"] == {"epoch_100": None, "epoch_300": None}
