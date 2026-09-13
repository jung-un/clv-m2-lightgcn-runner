import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from clv_m5_category_frequency_value_model import (
    M5CategoryFrequencyValueLightGCN,
    build_category_frequency_features,
)
import lightgcn_clv_m5_category_frequency_value_screen as runner


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


def _train_frame():
    return pd.DataFrame(
        [
            {"u_idx": 0, "i_idx": 0, "cat_idx": 0, "b_raw": "a"},
            {"u_idx": 0, "i_idx": 1, "cat_idx": 1, "b_raw": "a"},
            {"u_idx": 0, "i_idx": 0, "cat_idx": 0, "b_raw": "b"},
            {"u_idx": 1, "i_idx": 2, "cat_idx": 1, "b_raw": "c"},
        ]
    )


def _features():
    return build_category_frequency_features(
        _train_frame(),
        n_users=2,
        n_items=3,
        n_categories=2,
        q_n=np.array([0.8, 0.4], dtype=np.float32),
        clv_valid=np.array([True, True]),
        shrinkage_strength=0.0,
    )


def _model(*, rho_category_n=0.05, layers=1):
    torch.manual_seed(17)
    return M5CategoryFrequencyValueLightGCN(
        n_users=2,
        n_items=3,
        n_categories=2,
        user_q_v=np.array([0.2, 0.8], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_price_percentile=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        item_price_valid=np.array([True, True, True]),
        category_features=_features(),
        adj=_adj(),
        id_dim=4,
        category_dim=2,
        rho_value=0.05,
        rho_category_n=rho_category_n,
        n_layers=layers,
        pref_reg=1e-4,
        basis_bandwidth=0.25,
    )


def test_category_frequency_decomposes_qn_by_basket_not_item_rows():
    features = _features()

    np.testing.assert_allclose(
        features.user_category_n_residual,
        [[0.2, -0.2], [-0.2, 0.2]],
        atol=1e-7,
    )
    np.testing.assert_array_equal(features.item_category, [0, 1, 1])
    assert features.diagnostics["n_transactions"] == 3
    assert features.diagnostics["uncentered_category_n_sum_max_error"] < 1e-7
    assert features.diagnostics["centered_category_n_row_sum_max_abs"] < 1e-7


def test_invalid_user_category_n_is_zero():
    features = build_category_frequency_features(
        _train_frame(),
        n_users=2,
        n_items=3,
        n_categories=2,
        q_n=np.array([0.8, 0.4], dtype=np.float32),
        clv_valid=np.array([True, False]),
        shrinkage_strength=10.0,
    )
    np.testing.assert_allclose(features.user_category_n_residual[1], 0.0)


def test_value_and_category_n_blocks_share_one_lightgcn():
    model = _model(layers=1)
    user0, item0 = model.layer0_embeddings()
    full0 = torch.cat([user0, item0], dim=0)
    expected = (full0 + torch.sparse.mm(model.adj, full0)) / 2.0

    user, item = model.propagated_embeddings()
    torch.testing.assert_close(user, expected[:2])
    torch.testing.assert_close(item, expected[2:])
    assert model.total_dim == 9
    assert model.representation_diagnostics()["explicit_q_n_in_m2"] is True


def test_category_n_off_arm_has_exactly_zero_category_score():
    model = _model(rho_category_n=0.0, layers=1)
    components = model.candidate_score_components(
        torch.tensor([0, 1]), torch.tensor([2, 0])
    )
    torch.testing.assert_close(
        components["category_n"], torch.zeros_like(components["category_n"])
    )
    assert model.representation_diagnostics()["explicit_q_n_in_m2"] is False


def test_same_weighted_ranking_path_updates_id_and_category_basis():
    model = _model(layers=1)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[2, 1], [0, 1]])
    user, item = model.propagated_embeddings()
    positive = (user[users] * item[positives]).sum(1)
    negative = (user[users, None] * item[negatives]).sum(2)
    row_weights = torch.tensor([0.9, 1.1])
    loss = (
        row_weights
        * torch.nn.functional.softplus(negative - positive[:, None]).mean(1)
    ).mean()
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.category_basis.weight.grad.abs().sum() > 0


def test_screen_has_only_same_run_value_only_and_category_n_arms(tmp_path):
    cfg = runner.configure_category_frequency_value_screen(
        out_dir=str(tmp_path / "results"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    prepared = {"marker": "same"}
    specs = runner.arm_specifications(prepared, cfg)
    summary = runner.preflight_summary(cfg)

    assert [spec["model_id"] for spec in specs] == list(runner.MODEL_IDS)
    assert [spec["rho_category_n"] for spec in specs] == [0.0, 0.05]
    assert all(spec["rho"] == 0.05 for spec in specs)
    assert all(spec["weighted"] is True for spec in specs)
    assert all(spec["assignment"] is prepared for spec in specs)
    assert summary["trained_models"] == list(runner.MODEL_IDS)
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["m3_edge_weight"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"epochs": 50},
        {"rho_category_n": 0.1},
        {"category_dim": 8},
        {"negative_count": 1},
    ],
)
def test_screen_rejects_unplanned_overrides(tmp_path, override):
    with pytest.raises(ValueError, match="카테고리별 N 최소 M5 screen"):
        runner.configure_category_frequency_value_screen(
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


def test_reading_requires_both_top10_economic_metrics():
    rows = {
        runner.VALUE_ONLY_M5_ID: _metrics(hit=1.0, vndcg=1.0),
        runner.CATEGORY_N_M5_ID: _metrics(hit=1.1, vndcg=1.01),
    }
    assert runner.screening_reading(rows)["category_n_directional_pass"] is True

    rows[runner.CATEGORY_N_M5_ID]["vndcg@10"] = 0.99
    assert runner.screening_reading(rows)["category_n_directional_pass"] is False


def test_colab_runs_category_frequency_value_screen_once():
    notebook = json.loads(
        Path("clv_m5_category_frequency_value_dunnhumby_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_category_frequency_value_screen(cfg)") == 1
    assert "historical_development_days_684_690" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "TO_BE_PINNED" not in source
