import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _adj(n_users=3, n_items=3):
    edges = [(0, 0), (1, 1), (2, 2)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices,
        values,
        (n_users + n_items,) * 2,
        check_invariants=False,
    ).coalesce()


def test_constant_gate_is_qv_only_and_keeps_invalid_users_inactive():
    from clv_m5_n_conditioned_value_basis_model import (
        M5NConditionedValueBasisLightGCN,
    )

    model = M5NConditionedValueBasisLightGCN(
        n_users=3,
        n_items=3,
        user_q_n=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        user_q_v=np.array([0.2, 0.5, 0.8], dtype=np.float32),
        user_clv_valid=np.array([True, False, True]),
        item_price_percentile=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        item_price_valid=np.array([True, True, True]),
        adj=_adj(),
        id_dim=4,
        rho=0.05,
        n_layers=1,
        pref_reg=1e-4,
        gate_delta=0.25,
        basis_bandwidth=0.25,
        constant_gate=1.25,
    )

    torch.testing.assert_close(
        model.n_gate(), torch.tensor([1.25, 0.0, 1.25])
    )
    assert "gate_offset_parameter" not in dict(model.named_parameters())
    assert "gate_slope_parameter" not in dict(model.named_parameters())
    diagnostics = model.representation_diagnostics()
    assert diagnostics["explicit_q_n_in_m2"] is False
    assert diagnostics["explicit_q_v_in_m2"] is True
    assert diagnostics["n_gate_mode"] == "fixed_constant"
    assert diagnostics["constant_gate"] == 1.25


def test_constant_gate_must_stay_inside_the_predeclared_gate_range():
    from clv_m5_n_conditioned_value_basis_model import (
        M5NConditionedValueBasisLightGCN,
    )

    with pytest.raises(ValueError, match="허용범위"):
        M5NConditionedValueBasisLightGCN(
            n_users=3,
            n_items=3,
            user_q_n=np.array([0.1, 0.5, 0.9], dtype=np.float32),
            user_q_v=np.array([0.2, 0.5, 0.8], dtype=np.float32),
            user_clv_valid=np.array([True, True, True]),
            item_price_percentile=np.array([0.1, 0.5, 0.9], dtype=np.float32),
            item_price_valid=np.array([True, True, True]),
            adj=_adj(),
            id_dim=4,
            rho=0.05,
            n_layers=1,
            pref_reg=1e-4,
            gate_delta=0.25,
            basis_bandwidth=0.25,
            constant_gate=1.3,
        )


def test_joint_nv_shuffle_preserves_tuples_inside_degree_bins():
    import lightgcn_clv_m5_n_conditioned_value_basis_controls as runner

    prepared = {
        "degree_bin": np.array([0, 0, 0, 1, 1, 1]),
        "q_n": np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8], dtype=np.float32),
        "q_v": np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2], dtype=np.float32),
        "clv_valid": np.array([True, True, False, True, False, True]),
    }
    shuffled = runner.degree_matched_nv_shuffle(
        prepared, seed=42, degree_bins=2
    )
    source = shuffled["source_user"]

    assert np.all(prepared["degree_bin"][source] == prepared["degree_bin"])
    assert np.all(source != np.arange(len(source)))
    original_tuples = sorted(
        zip(
            prepared["q_n"],
            prepared["q_v"],
            prepared["clv_valid"],
            strict=True,
        )
    )
    shuffled_tuples = sorted(
        zip(
            shuffled["q_n"],
            shuffled["q_v"],
            shuffled["clv_valid"],
            strict=True,
        )
    )
    assert shuffled_tuples == original_tuples


def test_control_runner_trains_only_nv_shuffle_and_qv_only(tmp_path):
    import lightgcn_clv_m5_n_conditioned_value_basis_controls as runner

    cfg = runner.configure_value_basis_controls(
        out_dir=str(tmp_path / "out"),
        actual_result_json=str(tmp_path / "actual.json"),
    )
    prepared = {
        "m2_shuffle": {"name": "shuffle"},
        "m2_v_only": {"name": "v_only"},
    }
    specs = runner.arm_specifications(prepared, cfg)
    summary = runner.preflight_summary(cfg)

    assert [spec["model_id"] for spec in specs] == list(runner.TRAINED_MODEL_IDS)
    assert all(spec["weighted"] is True for spec in specs)
    assert all(spec["assignment"] is prepared for spec in specs)
    assert all(spec["assignment_name"] == "observed_m4" for spec in specs)
    assert specs[0]["m2_assignment"] is prepared["m2_shuffle"]
    assert specs[0]["constant_gate"] is None
    assert specs[1]["m2_assignment"] is prepared["m2_v_only"]
    assert specs[1]["constant_gate"] == 1.25
    assert summary["trained_models"] == list(runner.TRAINED_MODEL_IDS)
    assert summary["reused_models"] == list(runner.REUSED_MODEL_IDS)
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False


def _metrics(*, accuracy, hit, vndcg):
    return {
        "recall@10": accuracy,
        "ndcg@10": accuracy,
        "recall@20": accuracy,
        "ndcg@20": accuracy,
        "recall@50": accuracy,
        "ndcg@50": accuracy,
        "price_purchase_amount_weighted_hit@10": hit,
        "vndcg@10": vndcg,
    }


def test_mechanism_reading_requires_both_strict_two_metric_comparisons():
    import lightgcn_clv_m5_n_conditioned_value_basis_controls as runner

    rows = {
        runner.M4_MODEL_ID: _metrics(accuracy=1.00, hit=1.00, vndcg=1.00),
        runner.ACTUAL_M5_MODEL_ID: _metrics(
            accuracy=1.01, hit=1.04, vndcg=1.04
        ),
        runner.SHUFFLED_M5_MODEL_ID: _metrics(
            accuracy=1.00, hit=1.03, vndcg=1.03
        ),
        runner.V_ONLY_M5_MODEL_ID: _metrics(
            accuracy=1.00, hit=1.02, vndcg=1.02
        ),
    }
    reading = runner.mechanism_reading(rows)

    assert reading["mechanism_screen_pass"] is True
    assert reading["actual_beats_degree_matched_nv_shuffle"] is True
    assert reading["actual_beats_qv_only_constant_gate"] is True
    assert reading["classification"] == "n_and_v_assignment_candidate"

    rows[runner.V_ONLY_M5_MODEL_ID]["vndcg@10"] = 1.05
    reading = runner.mechanism_reading(rows)
    assert reading["mechanism_screen_pass"] is False
    assert reading["actual_beats_degree_matched_nv_shuffle"] is True
    assert reading["actual_beats_qv_only_constant_gate"] is False
    assert reading["classification"] == "value_assignment_without_n_increment"


def test_colab_trains_only_two_controls_once_without_test_or_holdout():
    notebook = json.loads(
        Path(
            "clv_m5_n_conditioned_value_basis_controls_dunnhumby_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_value_basis_controls(cfg)") == 1
    assert "REUSED_MODEL_IDS" in source
    assert "TRAINED_MODEL_IDS" in source
    assert "08e0022881c822dc106e9c232c41c294025db27f" in source
    assert "TO_BE_PINNED" not in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "degree-matched N/V 순열" in source
    assert "V-only 상수 게이트" in source
