import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from clv_m3_repeat_frequency_first_hop import (
    build_repeat_frequency_first_hop,
)
from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)
from clv_m5_value_basis_repeat_frequency_first_hop_model import (
    M5ValueBasisRepeatFrequencyFirstHopLightGCN,
)
import lightgcn_clv_m5_value_basis_repeat_frequency_first_hop_screen as runner


def _adj(n_users=2, n_items=4):
    edges = [(0, 0), (0, 1), (1, 2), (1, 3)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    raw_values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(
        indices,
        raw_values,
        (n_users + n_items,) * 2,
        check_invariants=False,
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    values = raw_values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(
        indices, values, raw.shape, check_invariants=False
    ).coalesce()


def _blocks(adj, n_users=2, n_items=4):
    indices = adj.indices()
    values = adj.values()
    mask = (indices[0] < n_users) & (indices[1] >= n_users)
    users = indices[0, mask]
    items = indices[1, mask] - n_users
    base = torch.sparse_coo_tensor(
        torch.stack([users, items]),
        values[mask],
        (n_users, n_items),
        check_invariants=False,
    ).coalesce()
    return base, base.transpose(0, 1).coalesce()


def _train():
    return pd.DataFrame(
        [
            {"u_idx": 0, "i_idx": 0, "b_raw": "a"},
            {"u_idx": 0, "i_idx": 0, "b_raw": "b"},
            {"u_idx": 0, "i_idx": 0, "b_raw": "c"},
            {"u_idx": 0, "i_idx": 1, "b_raw": "d"},
            {"u_idx": 1, "i_idx": 2, "b_raw": "e"},
            {"u_idx": 1, "i_idx": 3, "b_raw": "f"},
        ]
    )


def _graph():
    adj = _adj()
    base, _ = _blocks(adj)
    indices = base.indices().numpy()
    return build_repeat_frequency_first_hop(
        _train(),
        indices[0],
        indices[1],
        base.values().numpy(),
        np.array([1.0, 0.5]),
        np.array([True, True]),
        n_users=2,
        target_strength=0.075,
        beta_cap=20.0,
    )


def _model(*, active=None):
    adj = _adj()
    base, item_user = _blocks(adj)
    if active is None:
        active = base
    torch.manual_seed(11)
    return M5ValueBasisRepeatFrequencyFirstHopLightGCN(
        n_users=2,
        n_items=4,
        user_q_n=np.array([1.0, 0.5], dtype=np.float32),
        user_q_v=np.array([0.2, 0.8], dtype=np.float32),
        user_q_c=np.array([0.4, 0.9], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_price_percentile=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
        item_price_valid=np.ones(4, dtype=bool),
        adj=adj,
        base_user_from_item=base,
        base_item_from_user=item_user,
        active_user_from_item=active,
        id_dim=4,
        rho=0.05,
        n_layers=2,
        pref_reg=1e-4,
        basis_bandwidth=0.25,
    )


def test_qn_reallocates_mass_from_more_repeated_to_less_repeated_edge():
    graph = _graph()
    ratio = graph.adjusted_coefficients / graph.base_coefficients

    assert graph.edge_repeat_count.tolist() == [3.0, 1.0, 1.0, 1.0]
    assert ratio[0] < 1.0 < ratio[1]
    np.testing.assert_allclose(ratio[2:], 1.0, atol=1e-7)
    for user in range(2):
        mask = graph.edge_users == user
        np.testing.assert_allclose(
            graph.adjusted_coefficients[mask].sum(),
            graph.base_coefficients[mask].sum(),
            atol=1e-7,
        )
    assert graph.diagnostics["target_reached"] is True
    assert graph.diagnostics["repeat_count_vs_coefficient_ratio_spearman"] < 0


def test_invalid_user_must_have_zero_qn():
    adj = _adj()
    base, _ = _blocks(adj)
    indices = base.indices().numpy()
    with pytest.raises(ValueError, match="invalid users must have q_N=0"):
        build_repeat_frequency_first_hop(
            _train(),
            indices[0],
            indices[1],
            base.values().numpy(),
            np.array([1.0, 0.5]),
            np.array([True, False]),
            n_users=2,
        )


def test_m3_off_path_matches_standard_qv_only_lightgcn():
    isolated = _model()
    torch.manual_seed(99)
    reference = M5NConditionedValueBasisLightGCN(
        n_users=2,
        n_items=4,
        user_q_n=np.array([1.0, 0.5], dtype=np.float32),
        user_q_v=np.array([0.2, 0.8], dtype=np.float32),
        user_q_c=np.array([0.4, 0.9], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_price_percentile=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
        item_price_valid=np.ones(4, dtype=bool),
        adj=_adj(),
        id_dim=4,
        rho=0.05,
        n_layers=2,
        pref_reg=1e-4,
        basis_bandwidth=0.25,
        constant_gate=1.0,
    )
    isolated.load_state_dict(reference.state_dict(), strict=True)
    actual = isolated.propagated_embeddings()
    expected = reference.propagated_embeddings()
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    diagnostics = isolated.representation_diagnostics()
    assert diagnostics["explicit_q_n_in_m2"] is False
    assert diagnostics["m3_user_first_hop_only"] is True


def test_m3_changes_user_first_hop_but_preserves_item_and_two_hop_paths():
    graph = _graph()
    base, item_user = _blocks(_adj())
    active = torch.sparse_coo_tensor(
        base.indices(),
        torch.from_numpy(graph.adjusted_coefficients),
        base.shape,
        check_invariants=False,
    ).coalesce()
    model = _model(active=active)
    user0, item0 = model.layer0_embeddings()

    user1_base = torch.sparse.mm(base, item0)
    user1_active = torch.sparse.mm(active, item0)
    item1_base = torch.sparse.mm(item_user, user0)
    user2_base = torch.sparse.mm(base, item1_base)
    item2_base = torch.sparse.mm(item_user, user1_base)
    user, item = model.propagated_embeddings()

    assert not torch.allclose(user1_active, user1_base)
    torch.testing.assert_close(user, (user0 + user1_active + user2_base) / 3.0)
    torch.testing.assert_close(item, (item0 + item1_base + item2_base) / 3.0)


def test_same_weighted_ranking_loss_updates_id_embeddings():
    model = _model()
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[2, 3], [0, 1]])
    user, item = model.propagated_embeddings()
    positive = (user[users] * item[positives]).sum(1)
    negative = (user[users, None] * item[negatives]).sum(2)
    weights = torch.tensor([0.9, 1.1])
    loss = (
        weights
        * torch.nn.functional.softplus(negative - positive[:, None]).mean(1)
    ).mean()
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0


def test_screen_has_only_matched_m3_off_and_qn_m3_arms(tmp_path):
    cfg = runner.configure_value_basis_repeat_frequency_screen(
        out_dir=str(tmp_path / "result"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    prepared = {"marker": "same"}
    specs = runner.arm_specifications(prepared, cfg)
    summary = runner.preflight_summary(cfg)

    assert [spec["model_id"] for spec in specs] == list(runner.MODEL_IDS)
    assert [spec["m3_arm"] for spec in specs] == ["off", "n_repeat"]
    assert all(spec["weighted"] is True for spec in specs)
    assert all(spec["assignment"] is prepared for spec in specs)
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["train_pairs_excluded_from_evaluation"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["m2"]["input"].startswith("observed q_V only")


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"epochs": 50},
        {"rho_value": 0.1},
        {"m3_target_strength": 0.1},
        {"negative_count": 1},
    ],
)
def test_screen_rejects_unplanned_overrides(tmp_path, override):
    with pytest.raises(ValueError, match="V-M2·N-M3·CLV-M4 최소 screen"):
        runner.configure_value_basis_repeat_frequency_screen(
            out_dir=str(tmp_path / "result"),
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
        runner.M3_OFF_MODEL_ID: _metrics(hit=1.0, vndcg=1.0),
        runner.M3_N_MODEL_ID: _metrics(hit=1.1, vndcg=1.01),
    }
    assert runner.screening_reading(rows)["n_m3_directional_pass"] is True

    rows[runner.M3_N_MODEL_ID]["vndcg@10"] = 0.99
    assert runner.screening_reading(rows)["n_m3_directional_pass"] is False


def test_colab_runs_two_arm_development_screen_once():
    notebook = json.loads(
        Path(
            "clv_m5_value_basis_repeat_frequency_first_hop_dunnhumby_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count(
        "result_df = run_value_basis_repeat_frequency_screen(cfg)"
    ) == 1
    assert "historical_development_days_684_690" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['fixed']['holdout_constructed'] is False" in source
    assert "d5d7462ec459a6e67d160ed4b853afb7b0ebfcb3" in source
    assert "TO_BE_PINNED" not in source
