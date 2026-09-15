import json
from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m5_economic_positive_weight_test as base
import lightgcn_clv_m5_nv_economic_positive_weight_k1 as screen
import lightgcn_clv_m5_nv_economic_positive_weight_k1_test as runner


def test_k1_config_locks_only_requested_ten_seed_protocol(tmp_path):
    cfg = runner.configure_m5_nv_economic_positive_k1_test_run(
        out_dir=str(tmp_path / "results")
    )

    assert cfg.dataset == "dunnhumby"
    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.negative_count == 1
    assert cfg.epochs == 100
    assert cfg.id_dim == 64
    assert cfg.n_layers == 2
    assert cfg.batch_size == 8192
    assert cfg.lr == 5e-4
    assert cfg.pref_reg == 1e-3
    assert cfg.rho == 0.15
    assert cfg.positive_weight_lambda == 0.5
    assert cfg.reused_seed42_json == ""
    assert len(screen.MODEL_IDS) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("negative_count", 5),
        ("seeds", (42,)),
        ("epochs", 99),
        ("n_layers", 3),
        ("lr", 1e-3),
        ("batch_size", 1024),
    ],
)
def test_k1_config_rejects_unrequested_changes(tmp_path, field, value):
    overrides = {"out_dir": str(tmp_path / "results"), field: value}
    with pytest.raises(ValueError):
        runner.configure_m5_nv_economic_positive_k1_test_run(**overrides)


def test_four_arms_form_the_same_m2_by_m4_factorial():
    prepared = {"placeholder": True}
    cfg = base.M5EconomicPositiveTestConfig(negative_count=1)

    specs = screen.arm_specifications(prepared, cfg)

    assert [spec["model_id"] for spec in specs] == list(screen.MODEL_IDS)
    assert [(spec["rho"], spec["weighted"]) for spec in specs] == [
        (0.0, False),
        (0.15, False),
        (0.0, True),
        (0.15, True),
    ]
    assert all(spec["assignment"] is prepared for spec in specs)
    assert all(spec["assignment_name"] == "observed" for spec in specs)


def test_interaction_uses_matched_four_arm_difference_in_differences():
    metrics = {
        model_id: {
            "recall@10": 1.0,
            "ndcg@10": 1.0,
            "recall@20": 1.0,
            "ndcg@20": 1.0,
            "recall@50": 1.0,
            "ndcg@50": 1.0,
            "vndcg@10": 1.0,
            "price_purchase_amount_weighted_hit@10": 1.0,
        }
        for model_id in screen.MODEL_IDS
    }
    metrics[screen.M2_MODEL_ID]["vndcg@10"] = 1.02
    metrics[screen.M4P_MODEL_ID]["vndcg@10"] = 1.03
    metrics[screen.M5_MODEL_ID]["vndcg@10"] = 1.06

    row = screen.interaction_rows(metrics).set_index("metric").loc["vndcg@10"]

    assert row["m2_effect"] == pytest.approx(0.02)
    assert row["m4_effect"] == pytest.approx(0.03)
    assert row["m2_increment_given_m4"] == pytest.approx(0.03)
    assert row["interaction_effect"] == pytest.approx(0.01)


def test_test_runner_patches_generic_harness_and_retrains_seed42(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cfg):
        captured["screen"] = base.screen
        captured["models"] = base.MODEL_IDS
        captured["negative_count"] = cfg.negative_count
        captured["reused"] = base._load_reused_seed42_arms({}, cfg)
        return pd.DataFrame([{"ok": True}])

    monkeypatch.setattr(base, "run_m5_economic_positive_test", fake_run)
    cfg = runner.configure_m5_nv_economic_positive_k1_test_run(
        out_dir=str(tmp_path / "results")
    )

    result = runner.run_m5_nv_economic_positive_k1_test(cfg)

    assert bool(result.iloc[0]["ok"])
    assert captured["screen"] is screen
    assert captured["models"] == screen.MODEL_IDS
    assert captured["negative_count"] == 1
    assert captured["reused"] == []


def test_colab_runs_the_locked_four_arm_ten_seed_job():
    notebook_path = Path(
        "clv_m5_explicit_nv_personalized_economic_positive_weight_"
        "dunnhumby_test_multiseed_k1_colab.ipynb"
    )
    if not notebook_path.exists():
        pytest.skip("notebook is added after the source commit is pinned")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_m5_nv_economic_positive_k1_test(cfg)") == 1
    assert "seeds=FULL_SEEDS" in source
    assert "cfg.negative_count == 1" in source
    assert "len(summary['models']) == 4" in source
    assert "summary['protocol_status']" in source
    assert "TO_BE_PINNED" not in source
