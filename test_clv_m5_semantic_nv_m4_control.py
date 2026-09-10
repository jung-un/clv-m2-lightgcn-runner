import json
from pathlib import Path

import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_semantic_nv_m4_control as runner
import lightgcn_clv_m5_semantic_nv_single_screen as semantic


def _manifest():
    return {
        "transactions": {"path": "/x/tx.csv", "bytes": 10, "sha256": "a" * 64},
        "item_metadata": {"path": "/x/item.csv", "bytes": 20, "sha256": "b" * 64},
    }


def _actual_payload(cfg):
    metrics = {
        "recall@10": 0.11,
        "ndcg@10": 0.12,
        "recall@20": 0.21,
        "ndcg@20": 0.22,
        "recall@50": 0.31,
        "ndcg@50": 0.32,
        "vndcg@10": 0.14,
        "price_purchase_amount_weighted_hit@10": 0.15,
    }
    arm = {
        "model_id": runner.M5_MODEL_ID,
        "role": "single_actual_m5_pilot",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": cfg.rho,
        "positive_weight_lambda": cfg.positive_weight_lambda,
        "clv_assignment": "observed",
        "metrics": metrics,
    }
    return {
        "code_version": semantic.CODE_VERSION,
        "config": {
            key: getattr(cfg, key)
            for key in (
                "dataset",
                "seed",
                "time_cutoff",
                "evaluation_days",
                "epochs",
                "id_dim",
                "economic_dim",
                "economic_bins",
                "shrinkage_strength",
                "rho",
                "price_axis_budget",
                "scale_delta",
                "positive_weight_lambda",
                "n_layers",
                "negative_count",
                "batch_size",
                "lr",
                "pref_reg",
                "input_days",
            )
        },
        "input_manifest": _manifest(),
        "arm": arm,
        "absolute_rows": [{"model_id": runner.M5_MODEL_ID, **metrics}],
    }


def test_control_trains_only_matched_m4_and_reuses_actual_m5(tmp_path):
    cfg = runner.configure_semantic_nv_m4_control(
        out_dir=str(tmp_path / "out"),
        actual_m5_result_json=str(tmp_path / "actual.json"),
    )
    summary = runner.preflight_summary(cfg)
    spec = runner.arm_specification({}, cfg)

    assert summary["trained_models"] == [runner.M4_MODEL_ID]
    assert summary["reused_models"] == [runner.M5_MODEL_ID]
    assert spec["rho"] == 0.0
    assert spec["weighted"] is True
    assert spec["assignment_name"] == "observed"


def test_actual_result_contract_checks_same_input_and_training_settings(tmp_path):
    actual_path = tmp_path / "actual.json"
    cfg = runner.configure_semantic_nv_m4_control(
        out_dir=str(tmp_path / "out"),
        actual_m5_result_json=str(actual_path),
    )
    payload = _actual_payload(cfg)
    actual_path.write_text(json.dumps(payload), encoding="utf-8")
    prepared = {
        "input_hash": legacy.moe.manifest_hash(payload["input_manifest"]),
    }

    loaded = runner.load_actual_m5(cfg, prepared)

    assert loaded["arm"]["model_id"] == runner.M5_MODEL_ID


def test_incremental_rule_requires_both_economic_metrics_to_improve():
    m4 = {
        "recall@10": 0.10,
        "ndcg@10": 0.11,
        "recall@20": 0.20,
        "ndcg@20": 0.21,
        "recall@50": 0.30,
        "ndcg@50": 0.31,
        "vndcg@10": 0.12,
        "price_purchase_amount_weighted_hit@10": 0.13,
    }
    m5 = dict(m4)
    m5["vndcg@10"] += 0.01
    m5["price_purchase_amount_weighted_hit@10"] += 0.02

    positive = runner.incremental_reading(m4, m5)
    m5["vndcg@10"] = m4["vndcg@10"]
    nonpositive = runner.incremental_reading(m4, m5)

    assert positive["m2_increment_signal"] is True
    assert positive["next_step"] == "run joint CLV shuffle and degree control"
    assert nonpositive["m2_increment_signal"] is False
    assert "rho/beta" in nonpositive["next_step"]


def test_colab_runs_only_m4_control_and_reuses_actual_result():
    notebook_path = Path(
        "clv_m5_semantic_nv_personalized_positive_dunnhumby_m4_control_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "run_semantic_nv_m4_control(cfg)" in source
    assert "m5_semantic_nv_single_426685be4486.json" in source
    assert "trained_models" in source
    assert "M4_MODEL_ID" in source
