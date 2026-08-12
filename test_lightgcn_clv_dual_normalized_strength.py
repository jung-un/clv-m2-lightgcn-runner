import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _curve_rows():
    rows = []
    for seed in (42, 43, 44):
        baseline = {
            "seed": seed,
            "model_id": "m1",
            "rho": 0.0,
            "revenue@10": 1.0,
            **{
                f"{metric}@{k}": 1.0
                for metric in ("recall", "ndcg")
                for k in (10, 20, 50)
            },
        }
        rows.append(baseline)
        for rho, main, shuffled, adapter in (
            (0.2, 1.02, 1.01, 1.015),
            (0.4, 1.05, 1.02, 1.03),
        ):
            for model_id, revenue in (
                ("dual_clv_fixed", main),
                ("dual_shuffled_user", shuffled),
                ("dual_adapter_only", adapter),
            ):
                rows.append(
                    {
                        **baseline,
                        "model_id": model_id,
                        "rho": rho,
                        "revenue@10": revenue,
                    }
                )
    return pd.DataFrame(rows)


def test_normalized_presets_are_validation_only_and_fixed(tmp_path):
    import lightgcn_clv_dual_normalized_strength as normalized

    cfg = normalized.configure_normalized_strength(
        "dunnhumby",
        tmp_path / "seed42.json",
        tmp_path / "multiseed.json",
        tmp_path / "controls.json",
    )
    assert cfg.seeds == (42, 43, 44)
    assert cfg.rho_grid == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert cfg.model_ids == (
        "dual_clv_fixed",
        "dual_shuffled_user",
        "dual_adapter_only",
    )
    assert (cfg.gate_shape, cfg.window_days) == ("equal", None)
    assert cfg.eval_test is cfg.eval_holdout is False


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": (42,)},
        {"rho_grid": (0.1, 0.2)},
        {"model_ids": ("dual_clv_fixed",)},
        {"eval_test": True},
        {"eval_holdout": True},
        {"gate_shape": "high"},
    ],
)
def test_normalized_config_rejects_protocol_changes(changes, tmp_path):
    import lightgcn_clv_dual_normalized_strength as normalized

    cfg = normalized.configure_normalized_strength(
        "dunnhumby",
        tmp_path / "seed42.json",
        tmp_path / "multiseed.json",
        tmp_path / "controls.json",
    )
    with pytest.raises(ValueError):
        normalized.validate_normalized_config(dataclasses.replace(cfg, **changes))


def test_equivalent_lambda_targets_same_effective_strength():
    from lightgcn_clv_dual_normalized_strength import equivalent_lambda

    assert equivalent_lambda(0.4, 0.2) == pytest.approx(2.0)
    assert equivalent_lambda(0.4, 0.8) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        equivalent_lambda(0.4, 0.0)


def test_seed_run_config_preserves_training_budget_but_stays_validation_only(tmp_path):
    import lightgcn_clv_dual_normalized_strength as normalized

    cfg = normalized.configure_normalized_strength(
        "dunnhumby",
        tmp_path / "seed42.json",
        tmp_path / "multiseed.json",
        tmp_path / "controls.json",
        out_dir=tmp_path / "out",
    )
    source = {
        field.name: field.default
        for field in dataclasses.fields(normalized.moe.MoEConfig)
    }
    source.update(max_epochs=77, encoder_epochs=66)
    run_cfg = normalized._run_cfg(cfg, {"config": source}, 43)

    assert run_cfg.seed_list == (43,)
    assert run_cfg.max_epochs == 77
    assert run_cfg.encoder_epochs == 66
    assert run_cfg.eval_test is run_cfg.eval_holdout is False


def test_load_model_takes_gate_shape_from_normalized_protocol(monkeypatch):
    import lightgcn_clv_dual_normalized_strength as normalized

    observed = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            observed["hidden_dim"] = kwargs["hidden_dim"]
            observed["expert_dim"] = kwargs["expert_dim"]

        def to(self, _device):
            return self

        def load_state_dict(self, state):
            observed["state"] = state

        def eval(self):
            return self

        def set_gate_shape(self, gate_shape):
            observed["gate_shape"] = gate_shape

        def set_eval_axes(self, axes):
            observed["axes"] = axes

    monkeypatch.setattr(normalized, "CLVDualAxisEmbeddingModel", FakeModel)
    monkeypatch.setattr(normalized.dual, "_fresh_base", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        normalized.torch,
        "load",
        lambda *_args, **_kwargs: {
            "baseline_state_hash": "baseline",
            "state": {"weight": 1},
        },
    )
    prepared = {
        "user_profile": object(),
        "item_profile": object(),
        "q_n": object(),
        "q_v": object(),
        "baseline_hash": "baseline",
    }
    run_cfg = SimpleNamespace(expert_hidden_dim=32, expert_dim=16)

    normalized._load_model(
        prepared,
        run_cfg,
        "high",
        seed=42,
        model_id="dual_clv_fixed",
        checkpoint=Path("adapter.pt"),
    )

    assert observed == {
        "hidden_dim": 32,
        "expert_dim": 16,
        "state": {"weight": 1},
        "gate_shape": "high",
        "axes": "n_plus_v",
    }


def test_normalized_decision_selects_best_common_rho_with_guards_and_controls():
    import lightgcn_clv_dual_normalized_strength as normalized

    decision = normalized.normalized_strength_decision(_curve_rows())

    assert decision["success"] is True
    assert decision["selected_rho"] == pytest.approx(0.4)
    assert decision["failed_conditions"] == []


def test_normalized_decision_rejects_capacity_explanation():
    import lightgcn_clv_dual_normalized_strength as normalized

    rows = _curve_rows()
    rows.loc[
        rows.model_id.eq("dual_adapter_only") & rows.rho.eq(0.4),
        "revenue@10",
    ] = 1.06
    decision = normalized.normalized_strength_decision(rows)

    assert decision["success"] is True
    assert decision["selected_rho"] == pytest.approx(0.2)


def test_normalized_colab_runs_both_datasets_without_training_or_protected_splits():
    notebook = json.loads(
        Path("clv_dual_normalized_strength_colab.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", source)
    assert "run_normalized_strength" in source
    assert "results_clv_dual_dunnhumby" in source
    assert "results_clv_dual_hm_w60" in source
    for forbidden in (
        "eval_test=True",
        "eval_holdout=True",
        "train_moe",
        "_train_variant",
        "run_experiment",
    ):
        assert forbidden not in source


def test_normalized_colab_clears_stale_project_modules_after_reclone():
    notebook = json.loads(
        Path("clv_dual_normalized_strength_colab.ipynb").read_text()
    )
    setup_source = "".join(notebook["cells"][1].get("source", []))

    assert "sys.modules" in setup_source
    assert "name.startswith('lightgcn_clv')" in setup_source
    assert setup_source.index("sys.modules") > setup_source.index("git', 'checkout")
