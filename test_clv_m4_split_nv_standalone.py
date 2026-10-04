import json

import numpy as np
import pandas as pd

import clv_m5_m3_m4_split_nv_screen as screen
import lightgcn_clv_m3_centered_value_graph as m3


def _row(model_id, value, *, recall50=None):
    row = {"model_id": model_id, "seed": 44, "epoch": 300}
    for metric in screen.ACCURACY + screen.ECONOMIC:
        row[metric] = float(value)
    if recall50 is not None:
        row["recall@50"] = float(recall50)
    return row


def test_standalone_m4_b_reuses_the_exact_m1_binary_adjacency():
    adjacency = object()
    graph = screen.binary_m4_graph({"data": {"adj": adjacency}})

    assert graph["adjacency"] is adjacency
    assert graph["beta"] == 0.0
    assert graph["audit"] == {"graph": "binary", "edge_weights_changed": False}


def test_standalone_spec_changes_only_the_m4_loss_on_the_binary_graph():
    spec = screen.standalone_m4_b_spec()

    assert spec["model_id"] == screen.ARM_M4_B
    assert spec["arm"] == "binary"
    assert spec["gamma"] == 0.0
    assert spec["stage"] == "m4_split_nv_standalone_dev"


def test_existing_seed43_result_accepts_seed_stored_in_config(tmp_path, monkeypatch):
    monkeypatch.setattr(screen.v3, "default_out_dir", lambda _: str(tmp_path / "results"))
    cfg = screen.configure(seed=43)
    root = tmp_path / "results_clv_m5_m3_m4_split_nv_s43_v1"
    root.mkdir()
    absolute = root / "absolute.csv"
    pd.DataFrame([
        {"model_id": model_id, "seed": 43, "epoch": epoch}
        for model_id in (m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY, screen.ARM_B)
        for epoch in (100, 300)
    ]).to_csv(absolute, index=False)
    stored = {
        "config": {
            key: getattr(cfg, key)
            for key in ("epochs", "eval_every", "batch_size", "lr", "n_layers",
                        "id_dim", "pref_reg", "negative_count")
        } | {"seeds": [43]},
        "input_hash": "same-input",
        "result_paths": {"absolute_csv": str(absolute)},
    }
    (root / f"{screen.CODE_VERSION}_legacy.json").write_text(json.dumps(stored))

    curve, _ = screen._existing_m5_b_curves(cfg, "same-input")

    assert set(curve.seed) == {43}


def test_factorial_reading_requires_m5_b_to_beat_both_single_interventions():
    curve = pd.DataFrame([
        _row(m3.M1_MODEL_ID, 1.00),
        _row(m3.ARM_VALUE_ACTIVITY, 1.04),
        _row(screen.ARM_M4_B, 1.06),
        _row(screen.ARM_B, 1.12),
    ])

    result = screen.factorial_reading(curve, seed=44)

    assert result["m5_b_economic_at10_above_both_standalones"] is True
    assert result["m5_b_accuracy_guard_99pct_vs_better_standalone"] is True
    assert result["combination_candidate"] is True
    assert np.isclose(
        result["interaction_absolute"]["price_purchase_amount_weighted_hit@10"],
        0.02,
    )


def test_factorial_reading_rejects_accuracy_loss_against_the_better_standalone():
    curve = pd.DataFrame([
        _row(m3.M1_MODEL_ID, 1.00),
        _row(m3.ARM_VALUE_ACTIVITY, 1.04),
        _row(screen.ARM_M4_B, 1.06),
        _row(screen.ARM_B, 1.12, recall50=1.00),
    ])

    result = screen.factorial_reading(curve, seed=44)

    assert result["m5_b_economic_at10_above_both_standalones"] is True
    assert result["m5_b_accuracy_guard_99pct_vs_better_standalone"] is False
    assert result["combination_candidate"] is False
