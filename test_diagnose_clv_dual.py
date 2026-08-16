import json
from pathlib import Path

import pandas as pd
import pytest


def _row(model_id, lam, revenue, strength, *, gate="high"):
    return {
        "seed": 42,
        "model_id": model_id,
        "split": "val",
        "gate_shape": gate,
        "lambda": lam,
        "effective_strength": strength,
        "revenue@10": revenue,
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
        "effective_n_ratio": 0.2,
        "effective_v_ratio": 0.3,
        "effective_total_ratio": strength / lam if lam else 0.5,
        "expert_score_corr": 0.1,
        "expert_top10_jaccard": 0.05,
        "저CLV_revenue@10": revenue * 0.8,
        "중CLV_revenue@10": revenue,
        "고CLV_revenue@10": revenue * 1.2,
    }


def test_diagnose_existing_results_finds_crossings_and_writes_outputs(tmp_path):
    from diagnose_clv_dual import diagnose_dual_results

    rows = [
        _row("m1", 0.0, 1.0, 0.0, gate="none"),
        _row("dual_clv_fixed", 0.5, 1.03, 0.2),
        _row("dual_clv_fixed", 1.0, 1.10, 0.4),
        _row("dual_clv_fixed", 2.0, 1.09, 0.8),
        _row("dual_shuffled_user", 0.5, 1.04, 0.1),
        _row("dual_shuffled_user", 1.0, 1.06, 0.3),
        _row("dual_shuffled_user", 2.0, 1.08, 0.5),
        _row("dual_adapter_only", 0.5, 1.02, 0.18),
        _row("dual_adapter_only", 1.0, 1.07, 0.36),
        _row("dual_adapter_only", 2.0, 1.10, 0.72),
    ]
    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "selected_operating_point": {
                    "gate_shape": "high",
                    "lambda": 1.0,
                    "revenue@10": 1.10,
                    "effective_strength": 0.4,
                },
                "screening_decision": {"success": False},
                "axis_preflight": {"q_n_q_v_corr": 0.4},
            }
        )
    )

    result = diagnose_dual_results(csv_path, json_path, tmp_path / "diagnostics")

    selected = result["selected_comparison"].set_index("comparison")
    assert selected.loc["M1", "absolute_delta"] == pytest.approx(0.10)
    assert selected.loc["dual_shuffled_user", "absolute_delta"] == pytest.approx(0.04)
    assert selected.loc["dual_adapter_only", "absolute_delta"] == pytest.approx(0.03)
    assert result["crossings"].query(
        "control == 'dual_shuffled_user'"
    ).iloc[0]["lambda_interval"] == "0.5→1"
    matched = result["matched_strength"].set_index("control")
    assert round(matched.loc["dual_shuffled_user", "interpolated_control_revenue"], 6) == 1.07
    dominance = result["same_lambda_dominance"]
    failed = dominance.loc[~dominance.primary_wins]
    assert set(zip(failed.control, failed["lambda"])) == {
        ("dual_shuffled_user", 0.5),
        ("dual_adapter_only", 2.0),
    }
    assert len(result["primary_gate_summary"]) == 1
    assert result["limitations"]["user_quadrant_available"] is False
    for key in (
        "curve_table_csv",
        "primary_gate_summary_csv",
        "same_lambda_dominance_csv",
        "matched_curve_dominance_csv",
        "selected_comparison_csv",
        "crossings_csv",
        "matched_strength_csv",
        "summary_json",
        "lambda_curve_png",
        "strength_curve_png",
    ):
        assert Path(result["paths"][key]).exists(), key


def test_diagnostic_colab_reads_existing_drive_results_without_training():
    notebook = json.loads(Path("clv_dual_diagnostic_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "run_experiment" not in source
    assert "RESULT_SPECS" in source
    assert "diagnose_dual_results" in source
    assert "results_clv_dual_dunnhumby" in source
    assert "results_clv_dual_hm_w60" in source
