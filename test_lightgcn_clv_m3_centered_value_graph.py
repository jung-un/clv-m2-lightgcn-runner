import json

import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m3_centered_value_graph as graph


def _cfg(**overrides):
    defaults = {"out_dir": "/tmp/centered_graph", "m1_result_dir": "/tmp/capacity"}
    return graph.configure_centered_graph(**(defaults | overrides))


def _train(n_users=200, n_items=60, seed=0):
    """Synthetic history where expensive items take a large share of every spend."""

    rng = np.random.default_rng(seed)
    price = np.linspace(1.0, 50.0, n_items)
    rows = []
    for user in range(n_users):
        items = rng.choice(n_items, size=rng.integers(6, 20), replace=False)
        for basket, item in enumerate(items):
            for repeat in range(int(rng.integers(1, 4))):
                rows.append({"u_idx": user, "i_idx": int(item),
                             "b_raw": f"{user}-{item}-{repeat}",
                             "v": float(price[item]) * float(rng.uniform(0.8, 1.2))})
    return pd.DataFrame(rows), price


def _signals(train, n_users=200, n_items=60):
    return graph.centered_edge_signals(train, n_users, n_items)


def test_double_centering_removes_both_margins():
    train, _ = _train()
    s = _signals(train)
    users, items = s["edge_users"], s["edge_items"]
    for name in ("z_value", "z_activity"):
        z = s[name]
        item_mean = np.bincount(items, weights=z, minlength=60) / np.maximum(
            np.bincount(items, minlength=60), 1)
        # the item margin is removed exactly; the customer margin is restored later
        # by normalising each customer's weights to mean one
        assert np.abs(item_mean).max() < 1e-9, name
        assert abs(float(z.mean())) < 1e-9, name


def test_balancing_flattens_both_margins_at_once():
    """Both sides average one, so neither customers nor items get globally louder."""

    train, _ = _train()
    s = _signals(train)
    q = np.linspace(0.1, 0.9, 200)
    w = graph.edge_weights(s, q, q, gamma=1.0, beta=0.4)

    user_count = np.bincount(s["edge_users"], minlength=200)
    item_count = np.bincount(s["edge_items"], minlength=60)
    user_mass = np.bincount(s["edge_users"], weights=w, minlength=200)
    item_mean = (np.bincount(s["edge_items"], weights=w, minlength=60)
                 / np.maximum(item_count, 1))
    seen_u, seen_i = user_count > 0, item_count > 0

    assert (np.abs(user_mass[seen_u] - user_count[seen_u])
            / user_count[seen_u]).max() < 1e-3
    assert np.abs(item_mean[seen_i] - 1.0).max() < 1e-3

    # 한쪽만 맞추면 상품 쪽 여백이 1에서 벗어난 채로 남는다
    one_round = graph._balance_margins(
        np.exp(0.4 * (q[s["edge_users"]] * s["z_value"])), s, rounds=1)
    single = (np.bincount(s["edge_items"], weights=one_round, minlength=60)
              / np.maximum(item_count, 1))
    assert np.abs(single[seen_i] - 1.0).max() > np.abs(item_mean[seen_i] - 1.0).max()


def test_gamma_zero_ignores_the_activity_axis():
    train, _ = _train()
    s = _signals(train)
    q_value, q_activity = np.linspace(0.1, 0.9, 200), np.linspace(0.9, 0.1, 200)
    without = graph.edge_weights(s, q_value, q_activity, 0.0, 0.4)
    other = graph.edge_weights(s, q_value, q_activity * 0.0, 0.0, 0.4)
    assert np.allclose(without, other)
    with_activity = graph.edge_weights(s, q_value, q_activity, 1.0, 0.4)
    assert not np.allclose(without, with_activity)


def test_beta_is_calibrated_to_the_target_spread():
    train, _ = _train()
    s = _signals(train)
    q = np.linspace(0.1, 0.9, 200)
    for target in (0.15, 0.20, 0.30):
        beta = graph.calibrate_beta(s, q, q, 1.0, target)
        w = graph.edge_weights(s, q, q, 1.0, beta)
        assert abs(w.std() / w.mean() - target) < 5e-3


def test_axes_keep_their_own_size_so_data_sets_the_ratio():
    """A dataset without repeat purchases contributes almost no activity signal."""

    train, _ = _train()
    flat = train.copy()
    flat["b_raw"] = [f"{u}-{i}" for u, i in zip(flat.u_idx, flat.i_idx)]   # 상품당 바구니 1개
    rich, poor = _signals(train), _signals(flat)

    assert rich["z_activity"].std() > 10 * poor["z_activity"].std()
    assert poor["z_activity"].std() < 0.05 * poor["z_value"].std()


def _prepared(train, price, n_users=200, n_items=60):
    return {
        "data": {"n_users": n_users, "n_items": n_items, "train": train},
        "item_amount_percentile": pd.Series(price).rank(pct=True).to_numpy(),
    }


def test_audit_reports_the_item_side_tilt_it_is_there_to_catch():
    """The gate exists because per-customer normalisation can re-tilt the item side."""

    train, price = _train()
    s = _signals(train)
    prepared = _prepared(train, price)
    users, items = s["edge_users"], s["edge_items"]

    def customer_only(raw):
        """이전 설계(고객 쪽만 평균 1)를 감사가 잡아내는지 보기 위한 재현"""
        mean = (np.bincount(users, weights=raw, minlength=200)
                / np.maximum(np.bincount(users, minlength=200), 1))
        return raw / np.maximum(mean[users], 1e-12)

    # 상품 가격에 따라 커지도록 만든 가중치는 감사에서 +1에 가깝게 잡혀야 한다
    rising = customer_only(np.exp(0.3 * pd.Series(price).rank(pct=True).to_numpy()[items]))
    assert graph.audit_weights(rising, s, prepared)["item_weight_vs_price"] > 0.8

    # 기존 설계(고객 내 중심화만)도 감사에서 큰 기울기로 잡힌다
    value = train.groupby(["u_idx", "i_idx"], sort=True).v.sum().to_numpy()
    share = value / np.bincount(users, weights=value, minlength=200)[users]
    x = np.log(share + 1e-9)
    x = x - (np.bincount(users, weights=x, minlength=200)
             / np.maximum(np.bincount(users, minlength=200), 1))[users]
    single = customer_only(np.exp(0.4 * x / x.std()))
    assert graph.audit_weights(single, s, prepared)["item_weight_vs_price"] > 0.5


def test_gate_stops_training_when_a_design_fails():
    cfg = _cfg()
    good = {"coefficient_of_variation": 0.20, "customer_weight_vs_degree": 0.01,
            "item_weight_vs_price": 0.10, "item_weight_vs_popularity": 0.05,
            "max_customer_mass_error": 1e-12}
    graph.check_gates(good, cfg, "arm")

    for key, value in (("item_weight_vs_price", 0.88),
                       ("customer_weight_vs_degree", 0.77),
                       ("coefficient_of_variation", 0.012),
                       ("max_customer_mass_error", 0.5)):
        with pytest.raises(RuntimeError):
            graph.check_gates({**good, key: value}, cfg, "arm")


def test_config_and_arms_are_fixed_before_running():
    summary = graph.preflight_summary(_cfg())
    assert summary["reported_epochs"] == [100, 300]
    assert 100 in summary["evaluated_at_epochs"]
    assert summary["axes_not_rescaled_to_each_other"] is True
    assert [a["gamma"] for a in summary["arms"]] == [0.0, 1.0]
    with pytest.raises(ValueError):
        _cfg(epochs=150)
    with pytest.raises(ValueError):
        _cfg(negative_count=5)


def _curve_payload(seed, model_id, values, epochs=(100, 300), **extra):
    return {"model_id": model_id, "seed": seed, "id_dim": 64, "pref_reg": 1e-3, **extra,
            "curve": [{"epoch": e, "loss": 0.1,
                       "metrics": {"recall@10": v, "ndcg@10": v, "recall@20": v,
                                   "ndcg@20": v, "recall@50": v, "ndcg@50": v,
                                   "price_purchase_amount_weighted_hit@50": v * 10,
                                   "고CLV_recall@50": v * 2}}
                      for e, v in zip(epochs, values)]}


def test_baseline_curves_are_reused_only_when_the_protocol_matches(tmp_path):
    cfg = _cfg(seeds=(42,), m1_result_dir=str(tmp_path))
    root = tmp_path / "arms" / "abc"
    root.mkdir(parents=True)
    grid = [e for e in range(25, 301, 25)]
    payload = _curve_payload(42, graph.M1_MODEL_ID, [0.01] * len(grid), epochs=grid)

    wrong = {**payload, "id_dim": 128}
    (root / f"baseline_{graph.M1_MODEL_ID}_s42.json").write_text(json.dumps(wrong))
    with pytest.raises(RuntimeError, match="찾지 못했습니다"):
        graph.load_baseline_curves(cfg)

    (root / f"baseline_{graph.M1_MODEL_ID}_s42.json").write_text(json.dumps(payload))
    assert len(graph.load_baseline_curves(cfg)[42]) == len(grid)


def test_reading_separates_the_activity_axis_contribution():
    cfg = _cfg(seeds=(42, 43, 44))
    arms, baselines = [], {}
    for seed in (42, 43, 44):
        baselines[seed] = _curve_payload(seed, graph.M1_MODEL_ID, [0.010, 0.012])["curve"]
        arms.append(_curve_payload(seed, graph.ARM_VALUE, [0.0105, 0.0126],
                                   arm="value_only", gamma=0.0))
        arms.append(_curve_payload(seed, graph.ARM_VALUE_ACTIVITY, [0.0104, 0.0125],
                                   arm="value_and_activity", gamma=1.0))
    curve = graph.curve_table(arms, baselines)
    difference = graph.difference_table(curve, [100, 300])
    reading = graph.centered_graph_reading(difference, cfg)

    value = reading[f"{graph.ARM_VALUE}@300"]
    assert value["weighted_hit_50"]["positive_seeds"] == 3
    assert value["accuracy_guard_99pct"] is True
    assert value["candidate"] is True
    # 활동축을 더하면 오히려 낮아지는 합성 사례
    assert reading["activity_axis_contribution"]["epoch_300"]["positive_seeds"] == 0
    assert reading["arm_selected"] is False


def test_difference_table_refuses_to_drop_a_seed_whose_baseline_is_missing():
    """A missing M1 curve must stop the table, not quietly shrink the seed count."""

    arms = [_curve_payload(42, graph.ARM_VALUE, [0.0105, 0.0126], arm="value_only", gamma=0.0)]
    complete = graph.curve_table(
        arms, {42: _curve_payload(42, graph.M1_MODEL_ID, [0.010, 0.012])["curve"]})
    assert len(graph.difference_table(complete, [100, 300])) == 2

    with pytest.raises(KeyError, match=graph.M1_MODEL_ID):
        graph.difference_table(graph.curve_table(arms, {}), [100, 300])


def test_only_the_named_seeds_get_a_new_baseline(monkeypatch):
    """The capacity search ran seed 42 only; the rest need explicit permission."""

    trained = []
    monkeypatch.setattr(graph.capacity, "_prepare", lambda cfg: {"cfg": cfg})
    monkeypatch.setattr(graph.capacity, "_run_arm",
                        lambda prepared, cfg, spec, seed: trained.append((spec["model_id"], seed)))

    assert graph.train_missing_baselines(_cfg(seeds=(42, 43, 44))) == []
    assert trained == []

    allowed = _cfg(seeds=(42, 43, 44), allow_baseline_training=("43:m1", "44:m1"))
    assert graph.train_missing_baselines(allowed) == [43, 44]
    assert trained == [(graph.M1_MODEL_ID, 43), (graph.M1_MODEL_ID, 44)]

    # 허용 목록에 있어도 이번 실행 seed가 아니면 학습하지 않는다
    trained.clear()
    assert graph.train_missing_baselines(
        _cfg(seeds=(42,), allow_baseline_training=("43:m1",))) == []
    assert trained == []


def test_a_baseline_trained_under_another_setting_is_refused(monkeypatch):
    monkeypatch.setattr(graph.capacity, "_prepare", lambda cfg: {"cfg": cfg})
    monkeypatch.setattr(graph.capacity, "_run_arm",
                        lambda prepared, cfg, spec, seed: None)
    with pytest.raises(RuntimeError, match="id_dim"):
        graph.train_missing_baselines(
            _cfg(seeds=(43,), id_dim=128, allow_baseline_training=("43:m1",)))
