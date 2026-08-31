import pandas as pd
import pytest

import lightgcn_clv_m3_clv_relation as runner


def test_screening_config_locks_protected_split_and_m1_settings(tmp_path):
    cfg = runner.configure_m3_clv_relation_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["SEED_LIST"] == [42]
    assert cfg["MIN_ITEM_INTER"] == 1
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"MIN_ITEM_INTER": 10},
        {"LOSS_MODE": "user"},
        {"NEG_MODE": "hard50"},
        {"EVAL_TEST": True},
        {"EVAL_HOLDOUT": True},
    ],
)
def test_screening_config_rejects_invariant_violation(tmp_path, override):
    with pytest.raises(ValueError):
        runner.configure_m3_clv_relation_dunnhumby_run(
            out_dir=str(tmp_path), **override
        )


def _row(model_id, revenue, arp, distinct, top10, precision=0.001):
    row = {
        "model_id": model_id,
        "split": "val",
        "revenue@10": revenue,
        "arp@10": arp,
        "n_distinct@10": distinct,
        "top10_share@10": top10,
        "mean_hits@10": 10 * precision,
        "hit_value@10": revenue / (10 * precision),
    }
    for metric in runner.ACCURACY_METRICS:
        row[metric] = 1.0
    return row


def test_screen_requires_m1_control_and_mechanism_guards():
    frame = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.RELATION_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.GATE_ID, 1.02, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.03, 0.25, 200, 0.40),
        ]
    )
    decision = runner.screening_decision(frame)
    assert decision["success"] is True
    assert decision["clv_as_mixture_gate"]["passes_screen"] is True
    assert decision["clv_in_edge_and_gate"]["passes_screen"] is True


def test_price_shift_fails_even_if_weighted_hit_increases():
    frame = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.RELATION_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.GATE_ID, 1.20, 0.30, 200, 0.40),
            _row(runner.ALLOCATED_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.20, 0.30, 200, 0.40),
        ]
    )
    decision = runner.screening_decision(frame)
    assert decision["success"] is False
    assert not decision["clv_as_mixture_gate"]["guards"][
        "recommended_price_percentile"
    ]


def test_shuffle_config_locks_original_candidate_and_degree_matched_control(
    tmp_path,
):
    cfg = runner.configure_m3_clv_relation_shuffle_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["GRAPH_MODE"] == "clv_allocated_relation_gate_shuffle"
    assert cfg["GRAPH_ALPHA"] == 0.075
    assert cfg["SEED_LIST"] == [42]
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False

    summary = runner.shuffle_preflight_summary(cfg)
    assert summary["actual_model"] == runner.ALLOCATED_GATE_ID
    assert summary["shuffle_model"] == runner.SHUFFLE_ID
    assert summary["shuffle"]["within"] == "binary user-degree deciles"
    assert summary["shuffle"]["recomputes_item_baseline"] is True


def test_shuffle_attribution_requires_actual_clv_to_beat_m1_and_shuffle():
    passing = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.03, 0.25, 200, 0.40),
            _row(runner.SHUFFLE_ID, 1.01, 0.25, 200, 0.40),
        ]
    )
    for metric in runner.ACCURACY_METRICS:
        passing.loc[passing.model_id.eq(runner.ALLOCATED_GATE_ID), metric] = 1.02
        passing.loc[passing.model_id.eq(runner.SHUFFLE_ID), metric] = 1.01

    decision = runner.shuffle_attribution_decision(passing)
    assert decision["clv_attribution_supported"] is True
    assert decision["six_metric_balance_actual_vs_m1"] > 1.0
    assert decision["six_metric_balance_actual_vs_shuffle"] > 1.0

    failing = passing.copy()
    for metric in runner.ACCURACY_METRICS:
        failing.loc[failing.model_id.eq(runner.SHUFFLE_ID), metric] = 1.03
    decision = runner.shuffle_attribution_decision(failing)
    assert decision["clv_attribution_supported"] is False
    assert decision["six_metric_balance_actual_vs_m1"] > 1.0
    assert decision["six_metric_balance_actual_vs_shuffle"] < 1.0


def test_shuffle_comparison_rejects_a_different_m1_baseline():
    source = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.03, 0.25, 200, 0.40),
        ]
    )
    current = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.SHUFFLE_ID, 1.01, 0.25, 200, 0.40),
        ]
    )
    combined = runner.compose_shuffle_comparison(source, current)
    assert combined.model_id.tolist() == [
        "m1_baseline",
        runner.ALLOCATED_GATE_ID,
        runner.SHUFFLE_ID,
    ]
    assert combined.role.tolist() == ["baseline", "model", "control"]

    mismatched = current.copy()
    mismatched.loc[mismatched.model_id.eq("m1_baseline"), "recall@10"] = 0.90
    with pytest.raises(ValueError, match="M1 baseline"):
        runner.compose_shuffle_comparison(source, mismatched)


def test_shuffle_run_reuses_frozen_actual_result_and_saves_attribution(
    tmp_path,
    monkeypatch,
):
    cfg = runner.configure_m3_clv_relation_shuffle_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    source = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.03, 0.25, 200, 0.40),
        ]
    )
    for metric in runner.ACCURACY_METRICS:
        source.loc[source.model_id.eq(runner.ALLOCATED_GATE_ID), metric] = 1.02
    source_path = runner.source_comparison_path(cfg)
    source_path.parent.mkdir(parents=True)
    source.to_csv(source_path, index=False)

    current = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.SHUFFLE_ID, 1.01, 0.25, 200, 0.40),
        ]
    )
    for metric in runner.ACCURACY_METRICS:
        current.loc[current.model_id.eq(runner.SHUFFLE_ID), metric] = 1.01

    def fake_run_mode(received_cfg, graph_mode):
        assert received_cfg == cfg
        assert graph_mode == "clv_allocated_relation_gate_shuffle"
        return current

    monkeypatch.setattr(runner, "_run_mode", fake_run_mode)
    monkeypatch.setattr(
        runner,
        "_native_result_paths",
        lambda: {"json": "native.json", "val_csv": "native_val.csv"},
    )

    result = runner.run_shuffle_attribution_experiment(cfg)
    assert result.attrs["attribution_decision"]["clv_attribution_supported"] is True
    assert result.attrs["source_actual"] == str(source_path)
    assert pd.read_csv(result.attrs["result_paths"]["csv"]).model_id.tolist() == [
        "m1_baseline",
        runner.ALLOCATED_GATE_ID,
        runner.SHUFFLE_ID,
    ]
