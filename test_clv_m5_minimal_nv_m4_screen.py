import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from clv_m5_minimal_nv_model import M5MinimalNVEconomicLightGCN
import lightgcn_clv_m5_minimal_nv_m4_screen as runner


def _adj(n_users=2, n_items=3):
    edges = [(0, 0), (0, 1), (1, 2)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(
        indices, values, (n_users + n_items,) * 2
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model(*, rho=0.05, layers=1):
    torch.manual_seed(17)
    return M5MinimalNVEconomicLightGCN(
        n_users=2,
        n_items=3,
        user_n_centered=np.array([-0.5, 0.8], dtype=np.float32),
        user_v_centered=np.array([0.4, -0.6], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_repeat_centered=np.array([-0.8, 0.2, 0.9], dtype=np.float32),
        item_price_centered=np.array([-0.7, 0.1, 0.8], dtype=np.float32),
        item_semantic_valid=np.array([True, True, True]),
        adj=_adj(),
        id_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
        scale_delta=0.25,
    )


def test_minimal_nv_has_exactly_two_semantic_coordinates():
    model = _model(layers=0)
    user, item = model.economic_coordinates()

    expected_user = torch.tensor([[-0.5, 0.4], [0.8, -0.6]])
    expected_item = torch.tensor([[-0.8, -0.7], [0.2, 0.1], [0.9, 0.8]])
    torch.testing.assert_close(user, expected_user)
    torch.testing.assert_close(item, expected_item)
    assert model.economic_dim == 2
    assert model.total_dim == 6


def test_minimal_nv_block_is_propagated_with_id_in_one_lightgcn():
    model = _model(layers=1)
    user0, item0 = model.layer0_embeddings()
    full = torch.cat([user0, item0], dim=0)
    expected = (full + torch.sparse.mm(model.adj, full)) / 2.0

    user, item = model.propagated_embeddings()
    torch.testing.assert_close(user, expected[:2])
    torch.testing.assert_close(item, expected[2:])
    assert model.representation_diagnostics()["economic_graph_propagation"] is True


def test_same_loss_updates_id_and_bounded_n_v_scales():
    model = _model(layers=1)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[1, 2], [0, 1]])
    user, item = model.propagated_embeddings()
    positive = (user[users] * item[positives]).sum(1)
    negative = (user[users, None] * item[negatives]).sum(2)
    loss = torch.nn.functional.softplus(negative - positive[:, None]).mean()
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.n_scale_parameter.grad.abs() > 0
    assert model.v_scale_parameter.grad.abs() > 0
    assert 0.75 <= float(model.n_scale().detach()) <= 1.25
    assert 0.75 <= float(model.v_scale().detach()) <= 1.25


def _train_frame():
    return pd.DataFrame(
        [
            {"u_idx": 0, "i_idx": 0, "cat_idx": 0, "v": 10.0},
            {"u_idx": 0, "i_idx": 0, "cat_idx": 0, "v": 11.0},
            {"u_idx": 1, "i_idx": 0, "cat_idx": 0, "v": 9.0},
            {"u_idx": 0, "i_idx": 1, "cat_idx": 0, "v": 2.0},
            {"u_idx": 1, "i_idx": 1, "cat_idx": 0, "v": 3.0},
            {"u_idx": 2, "i_idx": 2, "cat_idx": 1, "v": 20.0},
            {"u_idx": 2, "i_idx": 2, "cat_idx": 1, "v": 19.0},
        ]
    )


def test_train_only_inputs_center_qn_qv_and_build_item_repeat_propensity():
    built = runner.build_minimal_nv_inputs(
        _train_frame(),
        n_users=3,
        n_items=3,
        q_n=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        q_v=np.array([0.3, 0.7, 0.8], dtype=np.float32),
        clv_valid=np.array([True, False, True]),
        item_price_percentile=np.array([0.6, 0.2, 0.9], dtype=np.float32),
        item_price_valid=np.array([True, True, True]),
    )

    np.testing.assert_allclose(built["user_n_centered"], [-0.6, 0.0, 0.8])
    np.testing.assert_allclose(built["user_v_centered"], [-0.4, 0.0, 0.6])
    np.testing.assert_allclose(
        built["item_repeat_rate"], [0.5, 0.25, 2.0 / 3.0]
    )
    np.testing.assert_allclose(
        built["item_repeat_centered"], [1.0 / 3.0, -1.0 / 3.0, 1.0]
    )
    np.testing.assert_allclose(
        built["item_price_centered"], [0.2, -0.6, 0.8], atol=1e-7
    )


def test_representation_shuffle_moves_only_qn_qv_inside_degree_bins():
    prepared = {
        "degree_bin": np.array([0, 0, 1, 1]),
        "user_n_centered": np.array([-0.8, -0.2, 0.4, 0.9]),
        "user_v_centered": np.array([-0.6, 0.1, 0.5, 0.8]),
        "clv_valid": np.array([True, True, True, False]),
    }
    shuffled = runner.representation_degree_matched_shuffle(
        prepared, seed=42, degree_bins=2
    )

    for target, source in enumerate(shuffled["source_user"]):
        assert prepared["degree_bin"][target] == prepared["degree_bin"][source]
        assert (
            shuffled["user_n_centered"][target]
            == prepared["user_n_centered"][source]
        )
        assert (
            shuffled["user_v_centered"][target]
            == prepared["user_v_centered"][source]
        )
    np.testing.assert_array_equal(
        np.sort(shuffled["user_n_centered"]),
        np.sort(prepared["user_n_centered"]),
    )


def test_screen_has_m4_actual_and_representation_shuffle_only(tmp_path):
    cfg = runner.configure_minimal_nv_m4_screen(
        out_dir=str(tmp_path / "results"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    prepared = {
        "m2_actual": {"name": "actual"},
        "m2_shuffle": {"name": "shuffle"},
    }
    specs = runner.arm_specifications(prepared, cfg)
    summary = runner.preflight_summary(cfg)

    assert [spec["model_id"] for spec in specs] == list(runner.MODEL_IDS)
    assert [spec["rho"] for spec in specs] == [0.0, 0.05, 0.05]
    assert all(spec["weighted"] for spec in specs)
    assert specs[0]["m2_assignment"] is prepared["m2_actual"]
    assert specs[1]["m2_assignment"] is prepared["m2_actual"]
    assert specs[2]["m2_assignment"] is prepared["m2_shuffle"]
    assert all(spec["assignment_name"] == "observed_m4" for spec in specs)
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True
    assert summary["m2"]["economic_dim"] == 2
    assert summary["m2"]["economic_graph_propagation"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.15},
        {"epochs": 50},
        {"negative_count": 1},
        {"n_layers": 1},
    ],
)
def test_screen_rejects_unplanned_overrides(tmp_path, override):
    with pytest.raises(ValueError, match="최소 N/V M5 screen"):
        runner.configure_minimal_nv_m4_screen(
            out_dir=str(tmp_path / "results"),
            baseline_result_dir=str(tmp_path / "baseline"),
            **override,
        )


def _metrics(*, hit, vndcg, accuracy=1.0):
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


def test_reading_requires_actual_to_beat_m4_and_nv_shuffle_on_both_primaries():
    rows = {
        runner.M4_MODEL_ID: _metrics(hit=1.00, vndcg=1.00),
        runner.M5_MODEL_ID: _metrics(hit=1.02, vndcg=1.01),
        runner.M5_SHUFFLE_MODEL_ID: _metrics(hit=1.01, vndcg=1.005),
    }
    reading = runner.screening_reading(rows)
    assert reading["m2_increment_signal"] is True
    assert reading["m2_assignment_signal"] is True
    assert reading["positive_screen"] is True
    assert reading["accuracy_classification"] == "strict_improvement"

    rows[runner.M5_SHUFFLE_MODEL_ID]["vndcg@10"] = 1.02
    reading = runner.screening_reading(rows)
    assert reading["m2_assignment_signal"] is False
    assert reading["positive_screen"] is False


def test_reading_rejects_accuracy_geomean_below_point_995():
    rows = {
        runner.M4_MODEL_ID: _metrics(hit=1.00, vndcg=1.00),
        runner.M5_MODEL_ID: _metrics(hit=1.02, vndcg=1.01, accuracy=0.99),
        runner.M5_SHUFFLE_MODEL_ID: _metrics(hit=1.01, vndcg=1.005),
    }
    reading = runner.screening_reading(rows)
    assert reading["accuracy_geomean_ratio_vs_m4"] == pytest.approx(0.99)
    assert reading["accuracy_classification"] == "reject_below_0_995"
    assert reading["positive_screen"] is False


def test_colab_runs_minimal_nv_m4_screen_once():
    notebook = json.loads(
        Path("clv_m5_minimal_nv_m4_dunnhumby_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_minimal_nv_m4_screen(cfg)") == 1
    assert "historical_development_days_684_690" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "TO_BE_PINNED" not in source
