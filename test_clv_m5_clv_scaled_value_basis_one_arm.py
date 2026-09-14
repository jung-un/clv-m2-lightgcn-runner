import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _adj(n_users=2, n_items=3):
    edges = [(0, 0), (0, 1), (1, 2)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(
        indices,
        values,
        (n_users + n_items,) * 2,
        check_invariants=False,
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(
        indices, normalized, raw.shape, check_invariants=False
    ).coalesce()


def _prepared():
    return {
        "data": {"n_users": 2, "n_items": 3, "adj": _adj()},
        "q_n": np.array([0.1, 0.9], dtype=np.float32),
        "q_v": np.array([0.2, 0.8], dtype=np.float32),
        "q_c": np.array([0.3, 0.9], dtype=np.float32),
        "clv_valid": np.array([True, True]),
        "item_amount_percentile": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "item_economic_valid": np.array([True, True, True]),
    }


def _metrics(offset=0.0):
    return {
        "recall@10": 0.02 + offset,
        "ndcg@10": 0.03 + offset,
        "recall@20": 0.04 + offset,
        "ndcg@20": 0.05 + offset,
        "recall@50": 0.06 + offset,
        "ndcg@50": 0.07 + offset,
        "price_purchase_amount_weighted_hit@10": 0.4 + offset,
        "vndcg@10": 0.02 + offset,
        "coverage@10": 0.01 + offset,
    }


def test_one_arm_contract_has_no_external_result_dependency(tmp_path):
    import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as runner

    cfg = runner.configure_clv_scaled_value_basis_one_arm(
        out_dir=str(tmp_path / "out")
    )
    summary = runner.preflight_summary(cfg)
    spec = runner.arm_specification(_prepared(), cfg)

    assert summary["trained_models"] == [runner.MODEL_ID]
    assert summary["reused_models"] == []
    assert summary["comparison"]["missing_reference_file_can_block_run"] is False
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["m2"]["separate_q_n_gate"] is False
    assert spec["weighted"] is True
    assert spec["assignment_name"] == "observed_m4"


def test_model_uses_qc_as_strength_without_a_separate_qn_gate(tmp_path):
    import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as runner

    cfg = runner.configure_clv_scaled_value_basis_one_arm(
        out_dir=str(tmp_path / "out")
    )
    prepared = _prepared()
    model = runner._build_model(
        prepared, cfg, runner.arm_specification(prepared, cfg)
    )

    torch.testing.assert_close(
        model.value_strength().cpu(), torch.tensor([0.3, 0.9])
    )
    assert model.constant_gate == 1.0
    assert "gate_offset_parameter" not in dict(model.named_parameters())
    assert "gate_slope_parameter" not in dict(model.named_parameters())
    diagnostics = model.representation_diagnostics()
    assert diagnostics["explicit_q_n_in_m2"] is False
    assert diagnostics["explicit_q_v_in_m2"] is True
    assert diagnostics["q_c_in_m2"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"epochs": 50},
        {"rho": 0.15},
        {"negative_count": 1},
        {"basis_bandwidth": 0.5},
    ],
)
def test_one_arm_rejects_unplanned_overrides(tmp_path, override):
    import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as runner

    with pytest.raises(ValueError, match="q_C×q_V M5 screen"):
        runner.configure_clv_scaled_value_basis_one_arm(
            out_dir=str(tmp_path / "out"), **override
        )


def test_directional_reading_requires_m5_to_beat_both_prior_models():
    import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as runner

    rank = {"top10_order_changed_user_share_vs_trained_id_only": 0.2}
    reading = runner._directional_reading(_metrics(), rank)
    assert reading["directional_candidate"] is True

    metrics = _metrics()
    metrics["recall@10"] = 0.015
    reading = runner._directional_reading(metrics, rank)
    assert reading["directional_candidate"] is False
    assert reading["m5_beats_prior_m4_on_all_four_top10_metrics"] is False
    assert reading["official_factorial_or_clv_attribution_decision_permitted"] is False


def test_reference_comparison_is_explicitly_cross_run():
    import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as runner

    comparison = runner._reference_comparison(_metrics())
    assert len(comparison) == 2 * len(runner.CORE_METRICS)
    assert set(comparison["reference_result_id"]) == {
        runner.M1_REFERENCE_ID,
        runner.M4_REFERENCE_ID,
    }
    assert set(comparison["comparison_scope"]) == {
        "different_run_descriptive_reference"
    }


def test_colab_runs_exactly_one_m5_without_test_or_holdout():
    notebook = json.loads(
        Path(
            "clv_m5_clv_scaled_value_basis_one_arm_dunnhumby_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_clv_scaled_value_basis_one_arm(cfg)") == 1
    assert "TO_BE_PINNED" not in source
    assert "PINNED_SOURCE_COMMIT" in source
    assert "summary['trained_models'] == [MODEL_ID]" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "m1_reference_json" not in source
    assert "m4_reference_json" not in source
