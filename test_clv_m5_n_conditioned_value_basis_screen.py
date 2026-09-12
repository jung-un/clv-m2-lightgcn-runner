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
        indices, values, (n_users + n_items,) * 2, check_invariants=False
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(
        indices, normalized, raw.shape, check_invariants=False
    ).coalesce()


def _model(*, rho=0.05, layers=0):
    from clv_m5_n_conditioned_value_basis_model import (
        M5NConditionedValueBasisLightGCN,
    )

    torch.manual_seed(17)
    return M5NConditionedValueBasisLightGCN(
        n_users=2,
        n_items=3,
        user_q_n=np.array([0.2, 0.9], dtype=np.float32),
        user_q_v=np.array([0.15, 0.8], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_price_percentile=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        item_price_valid=np.array([True, True, True]),
        adj=_adj(),
        id_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
        gate_delta=0.25,
        basis_bandwidth=0.25,
    )


def test_fixed_value_basis_is_unit_norm_and_preserves_price_nearness():
    from clv_m5_n_conditioned_value_basis_model import fixed_value_basis

    basis = fixed_value_basis(
        np.array([0.1, 0.5, 0.9]),
        np.array([True, True, True]),
        bandwidth=0.25,
    )

    assert basis.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0, atol=1e-6)
    near = float(basis[0] @ basis[1])
    far = float(basis[0] @ basis[2])
    assert near > far


def test_invalid_rows_have_zero_value_basis():
    from clv_m5_n_conditioned_value_basis_model import fixed_value_basis

    basis = fixed_value_basis(
        np.array([0.1, 0.5]), np.array([True, False]), bandwidth=0.25
    )
    np.testing.assert_allclose(basis[1], 0.0)


def test_qn_is_only_a_bounded_user_gate_and_qv_sets_value_position():
    model = _model(layers=0)
    with torch.no_grad():
        model.gate_offset_parameter.zero_()
        model.gate_slope_parameter.fill_(10.0)

    gate = model.n_gate()
    detached_gate = gate.detach()
    assert 0.75 <= float(detached_gate.min()) <= float(detached_gate.max()) <= 1.25
    assert float(detached_gate[1]) > float(detached_gate[0])

    user, item = model.economic_coordinates()
    assert user.shape == (2, 3)
    assert item.shape == (3, 3)
    assert float((user[0] @ item[0]).detach()) > float((user[0] @ item[2]).detach())
    diagnostics = model.representation_diagnostics()
    assert diagnostics["explicit_q_n_in_m2"] is True
    assert diagnostics["explicit_q_v_in_m2"] is True
    assert diagnostics["item_n_or_item_clv_input"] is False


def test_id_and_value_basis_are_propagated_in_one_lightgcn():
    model = _model(layers=1)
    user0, item0 = model.layer0_embeddings()
    full0 = torch.cat([user0, item0], dim=0)
    expected = (full0 + torch.sparse.mm(model.adj, full0)) / 2.0

    user, item = model.propagated_embeddings()
    torch.testing.assert_close(user, expected[:2])
    torch.testing.assert_close(item, expected[2:])
    assert model.total_dim == 7


def test_same_bpr_path_updates_id_and_n_gate_parameters():
    model = _model(layers=1)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[2, 1], [0, 1]])
    user, item = model.propagated_embeddings()
    positive = (user[users] * item[positives]).sum(1)
    negative = (user[users, None] * item[negatives]).sum(2)
    loss = torch.nn.functional.softplus(negative - positive[:, None]).mean()
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.gate_offset_parameter.grad.abs() > 0
    assert model.gate_slope_parameter.grad.abs() > 0


def test_screen_trains_only_new_m2_and_new_m5(tmp_path):
    import lightgcn_clv_m5_n_conditioned_value_basis_screen as runner

    cfg = runner.configure_n_conditioned_value_basis_screen(
        out_dir=str(tmp_path / "results"),
        baseline_result_dir=str(tmp_path / "baseline"),
        m1_reference_json=str(tmp_path / "m1.json"),
        m4_reference_json=str(tmp_path / "m4.json"),
    )
    prepared = {
        "m2_actual": {"name": "actual"},
    }
    specs = runner.arm_specifications(prepared, cfg)
    summary = runner.preflight_summary(cfg)

    assert [spec["model_id"] for spec in specs] == list(runner.TRAINED_MODEL_IDS)
    assert [spec["weighted"] for spec in specs] == [False, True]
    assert [spec["rho"] for spec in specs] == [0.05, 0.05]
    assert all(spec["architecture"] == "basis" for spec in specs)
    assert all(spec["m2_assignment"] is prepared["m2_actual"] for spec in specs)
    assert [spec["assignment_name"] for spec in specs] == [
        "unweighted",
        "observed_m4",
    ]
    assert summary["trained_models"] == list(runner.TRAINED_MODEL_IDS)
    assert summary["reused_models"] == list(runner.REFERENCE_MODEL_IDS)
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["m2"]["item_n_or_item_clv_input"] is False
    assert summary["m2"]["economic_graph_propagation"] is True


def _reference_payload(cfg, manifest, *, code_version, model_id, role, weighted):
    metrics = _metrics(recall=1.0, ndcg=1.0, hit=1.0, vndcg=1.0)
    row = {
        "model_id": model_id,
        "role": role,
        "seed": 42,
        "split": "historical_development_days_684_690",
        "final_epoch": 100,
        "rho": 0.0,
        "id_dim": 64,
        "n_layers": 2,
        **metrics,
    }
    arm = {
        **row,
        "negative_count": 5,
        "hard_negative": False,
        "clv_assignment": "observed_m4" if weighted else "observed",
        "positive_weight_lambda": 0.5 if weighted else 0.0,
        "metrics": metrics,
    }
    return {
        "code_version": code_version,
        "source_revision": "fixture-revision",
        "config": {
            key: getattr(cfg, key)
            for key in (
                "dataset",
                "seed",
                "time_cutoff",
                "evaluation_days",
                "epochs",
                "id_dim",
                "n_layers",
                "negative_count",
                "batch_size",
                "lr",
                "pref_reg",
                "input_days",
            )
        },
        "input_manifest": manifest,
        "absolute_rows": [row],
        "arms": {model_id: arm},
    }


def test_reused_m1_and_m4_are_loaded_from_the_two_fixed_result_files(tmp_path):
    import lightgcn_clv_m5_n_conditioned_value_basis_screen as runner

    manifest = {"transactions": {"sha256": "same", "bytes": 123}}
    cfg = runner.configure_n_conditioned_value_basis_screen(
        out_dir=str(tmp_path / "results"),
        baseline_result_dir=str(tmp_path / "baseline"),
        m1_reference_json=str(tmp_path / "m1.json"),
        m4_reference_json=str(tmp_path / "m4.json"),
    )
    m1 = _reference_payload(
        cfg,
        manifest,
        code_version="m5-m2-m4-joint-historical-screen-v1",
        model_id=runner.M1_MODEL_ID,
        role="factorial_m1",
        weighted=False,
    )
    m4 = _reference_payload(
        cfg,
        manifest,
        code_version="m5-minimal-nv-personalized-positive-development-screen-v1",
        model_id=runner.M4_MODEL_ID,
        role="matched_m4_only_control",
        weighted=True,
    )
    Path(cfg.m1_reference_json).write_text(json.dumps(m1), encoding="utf-8")
    Path(cfg.m4_reference_json).write_text(json.dumps(m4), encoding="utf-8")
    references, provenance = runner.load_reused_references(cfg)

    assert set(references) == set(runner.REFERENCE_MODEL_IDS)
    assert all(item["reused_without_retraining"] for item in provenance.values())
    assert references[runner.M4_MODEL_ID]["metrics"]["recall@10"] == 1.0

    m4["absolute_rows"][0]["model_id"] = "wrong_model"
    Path(cfg.m4_reference_json).write_text(json.dumps(m4), encoding="utf-8")
    with pytest.raises(RuntimeError, match="절대지표 행"):
        runner.load_reused_references(cfg)


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.15},
        {"basis_bandwidth": 0.5},
        {"epochs": 50},
        {"negative_count": 1},
    ],
)
def test_screen_rejects_unplanned_overrides(tmp_path, override):
    import lightgcn_clv_m5_n_conditioned_value_basis_screen as runner

    with pytest.raises(ValueError, match="N-conditioned V-basis M5 screen"):
        runner.configure_n_conditioned_value_basis_screen(
            out_dir=str(tmp_path / "results"),
            baseline_result_dir=str(tmp_path / "baseline"),
            m1_reference_json=str(tmp_path / "m1.json"),
            m4_reference_json=str(tmp_path / "m4.json"),
            **override,
        )


def _metrics(*, recall, ndcg, hit, vndcg):
    return {
        "recall@10": recall,
        "ndcg@10": ndcg,
        "recall@20": recall,
        "ndcg@20": ndcg,
        "recall@50": recall,
        "ndcg@50": ndcg,
        "price_purchase_amount_weighted_hit@10": hit,
        "vndcg@10": vndcg,
    }


def test_reading_treats_m5_over_m4_as_main_directional_check():
    import lightgcn_clv_m5_n_conditioned_value_basis_screen as runner

    rows = {
        runner.M1_MODEL_ID: _metrics(recall=1.00, ndcg=1.00, hit=1.00, vndcg=1.00),
        runner.M4_MODEL_ID: _metrics(recall=1.01, ndcg=1.01, hit=1.01, vndcg=1.01),
        runner.BASIS_M2_MODEL_ID: _metrics(
            recall=0.99, ndcg=0.99, hit=1.01, vndcg=1.01
        ),
        runner.BASIS_M5_MODEL_ID: _metrics(
            recall=1.02, ndcg=1.02, hit=1.03, vndcg=1.03
        ),
    }
    reading = runner.screening_reading(
        rows,
        economic_score_ratios={
            runner.BASIS_M2_MODEL_ID: 0.011,
            runner.BASIS_M5_MODEL_ID: 0.012,
        },
    )
    assert reading["directional_screen_pass"] is True
    assert reading["m5_top10_economics_beats_reused_m4"] is True
    assert reading["m5_top10_accuracy_beats_reused_m4"] is True
    assert reading["m2_top10_accuracy_beats_reused_m1"] is False
    assert reading["intervention_operational"] is True
    assert reading["clv_assignment_tested"] is False

    reading = runner.screening_reading(
        rows,
        economic_score_ratios={
            runner.BASIS_M2_MODEL_ID: 0.011,
            runner.BASIS_M5_MODEL_ID: 0.009,
        },
    )
    assert reading["directional_screen_pass"] is False
    assert reading["intervention_operational"] is False


def test_colab_runs_screen_once_without_constructing_test_or_holdout():
    notebook = json.loads(
        Path(
            "clv_m5_n_conditioned_value_basis_dunnhumby_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count(
        "result_df = run_n_conditioned_value_basis_screen(cfg)"
    ) == 1
    assert "TRAINED_MODEL_IDS" in source
    assert "REFERENCE_MODEL_IDS" in source
    assert "list(MODEL_IDS)" not in source
    assert "historical_development_days_684_690" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "TO_BE_PINNED" not in source
