import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _adj(n_users=3, n_items=3):
    indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2]], dtype=torch.long
    )
    return torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.shape[1]),
        (n_users + n_items,) * 2,
        check_invariants=False,
    ).coalesce()


def test_recency_adjusted_activity_rewards_recent_repeat_activity():
    from clv_m2_n_proxy_candidates import recency_adjusted_activity

    result = recency_adjusted_activity(
        np.array([4.0, 4.0, 0.0]),
        np.array([90.0, 20.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    np.testing.assert_allclose(result, [0.036, 0.008, 0.0])
    assert result[0] > result[1]


def test_bgnbd_returns_finite_nonnegative_expected_counts():
    from clv_m2_n_proxy_candidates import fit_bgnbd_expected_count

    rng = np.random.default_rng(42)
    age = rng.integers(30, 366, 120).astype(np.float64)
    frequency = rng.poisson(0.02 * age).astype(np.float64)
    recency = np.where(
        frequency > 0,
        age * rng.beta(3.0, 1.0, size=len(age)),
        0.0,
    )
    result = fit_bgnbd_expected_count(
        frequency,
        np.minimum(recency, age),
        age,
        np.ones(len(age), dtype=bool),
        horizon=7.0,
    )

    assert result.expected_count.shape == age.shape
    assert np.isfinite(result.expected_count).all()
    assert np.all(result.expected_count >= 0.0)
    assert result.diagnostics["fit_success"] is True
    assert result.diagnostics["horizon_days"] == 7.0
    assert set(result.parameters) == {"r", "alpha", "a", "b"}


def test_no_offset_gate_learns_only_the_qn_slope():
    from clv_m5_n_conditioned_value_basis_model import (
        M5NConditionedValueBasisLightGCN,
    )

    model = M5NConditionedValueBasisLightGCN(
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
        learn_gate_offset=False,
    )

    assert "gate_offset_parameter" not in dict(model.named_parameters())
    assert "gate_slope_parameter" in dict(model.named_parameters())
    torch.testing.assert_close(model.n_gate(), torch.ones(3))
    users, items = model.propagated_embeddings()
    loss = -(users * items).sum()
    loss.backward()
    assert model.gate_slope_parameter.grad is not None
    assert model.representation_diagnostics()["n_gate_mode"] == (
        "learned_q_n_slope_without_offset"
    )


def test_four_arms_change_only_n_proxy_and_keep_plain_m2(tmp_path):
    import lightgcn_clv_m2_n_proxy_value_basis_screen as runner

    cfg = runner.configure_m2_n_proxy_screen(
        out_dir=str(tmp_path / "out"),
        baseline_result_dir=str(tmp_path / "baseline"),
        m1_reference_json=str(tmp_path / "m1.json"),
    )
    observed = np.array([True, True, False])
    prepared = {
        "q_n": np.array([0.2, 0.8, 0.0], dtype=np.float32),
        "q_v": np.array([0.3, 0.7, 0.0], dtype=np.float32),
        "clv_valid": observed,
        "n_proxy_candidates": {
            "repeat_rate": np.array([0.2, 0.8, 0.0], dtype=np.float32),
            "recency_adjusted_activity": np.array(
                [0.4, 0.6, 0.0], dtype=np.float32
            ),
            "bgnbd_expected_count": np.array(
                [0.1, 0.9, 0.0], dtype=np.float32
            ),
        },
    }
    specs = runner.arm_specifications(prepared, cfg)
    summary = runner.preflight_summary(cfg)

    assert [spec["model_id"] for spec in specs] == list(runner.TRAINED_MODEL_IDS)
    assert [spec["n_proxy"] for spec in specs] == [
        "qv_only",
        "repeat_rate",
        "recency_adjusted_activity",
        "bgnbd_expected_count",
    ]
    assert [spec["constant_gate"] for spec in specs] == [1.0, None, None, None]
    assert all(spec["weighted"] is False for spec in specs)
    assert all(spec["rho"] == 0.05 for spec in specs)
    assert all(spec["m2_assignment"]["q_v"] is prepared["q_v"] for spec in specs)
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["m3_edge_weight"] is False
    assert summary["fixed"]["m4_loss_weight"] is False
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


def test_reading_requires_total_and_incremental_two_metric_signal():
    import lightgcn_clv_m2_n_proxy_value_basis_screen as runner

    rows = {
        runner.M1_MODEL_ID: _metrics(accuracy=1.0, hit=1.0, vndcg=1.0),
        runner.QV_ONLY_MODEL_ID: _metrics(accuracy=1.0, hit=1.01, vndcg=1.01),
        runner.REPEAT_RATE_MODEL_ID: _metrics(
            accuracy=1.01, hit=1.02, vndcg=1.02
        ),
        runner.RECENCY_ACTIVITY_MODEL_ID: _metrics(
            accuracy=1.01, hit=1.03, vndcg=1.005
        ),
        runner.BGNBD_MODEL_ID: _metrics(accuracy=0.99, hit=0.99, vndcg=1.04),
    }

    reading = runner.candidate_reading(rows)

    assert reading["candidates_for_next_stage"] == [runner.REPEAT_RATE_MODEL_ID]
    assert reading["by_model"][runner.REPEAT_RATE_MODEL_ID][
        "development_candidate"
    ] is True
    assert reading["by_model"][runner.RECENCY_ACTIVITY_MODEL_ID][
        "n_increment_signal_vs_qv_only"
    ] is False
    assert reading["winner_selected_from_single_seed"] is False


@pytest.mark.parametrize("override", [{"seed": 43}, {"rho": 0.15}, {"epochs": 50}])
def test_screen_rejects_unplanned_overrides(tmp_path, override):
    import lightgcn_clv_m2_n_proxy_value_basis_screen as runner

    with pytest.raises(ValueError, match="M2 N-proxy screen"):
        runner.configure_m2_n_proxy_screen(
            out_dir=str(tmp_path / "out"),
            baseline_result_dir=str(tmp_path / "baseline"),
            m1_reference_json=str(tmp_path / "m1.json"),
            **override,
        )


def test_colab_runs_four_arm_screen_once_without_final_test_or_holdout():
    notebook = json.loads(
        Path("clv_m2_n_proxy_value_basis_dunnhumby_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_m2_n_proxy_screen(cfg)") == 1
    assert "TRAINED_MODEL_IDS" in source
    assert "historical_development_days_684_690" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "TO_BE_PINNED" not in source
