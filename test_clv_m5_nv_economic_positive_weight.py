import numpy as np
import pandas as pd
import torch

from clv_m5_nv_economic_positive_weight_model import M5NVEconomicLightGCN
import lightgcn_clv_m5_nv_economic_positive_weight as runner
import lightgcn_clv_m5_nv_economic_positive_weight_test as test_runner


def _adj(n_users=3, n_items=4):
    rows = [0, 3, 0, 4, 1, 4, 1, 5, 2, 6]
    cols = [3, 0, 4, 0, 4, 1, 5, 1, 6, 2]
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(
        indices, values, (n_users + n_items,) * 2
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model(gate=(0.0, 0.5, 1.0), rho=0.15):
    torch.manual_seed(9)
    return M5NVEconomicLightGCN(
        n_users=3,
        n_items=4,
        user_economic_input=np.array(
            [
                [0.8, -0.1, -0.1, 0.0, 0.2],
                [0.4, -0.1, 0.0, 0.1, 0.0],
                [-0.4, 0.2, 0.0, -0.1, -0.1],
            ],
            dtype=np.float32,
        ),
        user_economic_valid=np.ones(3, dtype=bool),
        user_activity_gate=np.asarray(gate, dtype=np.float32),
        item_economic_input=np.array(
            [
                [-0.8, 0.75, -0.25, -0.25, -0.25],
                [-0.2, -0.25, 0.75, -0.25, -0.25],
                [0.3, -0.25, -0.25, 0.75, -0.25],
                [0.9, -0.25, -0.25, -0.25, 0.75],
            ],
            dtype=np.float32,
        ),
        item_economic_valid=np.ones(4, dtype=bool),
        adj=_adj(),
        id_dim=6,
        economic_dim=4,
        rho=rho,
        n_layers=1,
        pref_reg=1e-4,
    )


def test_q_n_is_post_projection_strength_only():
    model = _model()
    user, _ = model.economic_coordinates()
    torch.testing.assert_close(user[0], torch.zeros_like(user[0]))

    ungated = _model(gate=(1.0, 1.0, 1.0))
    ungated.load_state_dict(model.state_dict())
    full, _ = ungated.economic_coordinates()
    torch.testing.assert_close(user[1], 0.5 * full[1])
    torch.testing.assert_close(user[2], full[2])


def test_rho_zero_is_exact_nonintervention():
    model = _model(rho=0.0)
    full_user, full_item = model.propagated_embeddings()
    id_user, id_item = model.id_embeddings()
    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def _train_frame():
    rows = []
    for item, amount in enumerate(range(1, 9)):
        rows.append(
            {
                "u_idx": item % 3,
                "i_idx": item,
                "cat_idx": item % 2,
                "v": float(amount),
            }
        )
    rows.append({"u_idx": 0, "i_idx": 7, "cat_idx": 1, "v": 8.0})
    return pd.DataFrame(rows)


def _inputs():
    return runner.build_nv_economic_inputs(
        _train_frame(),
        n_users=3,
        n_items=8,
        q_n=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        q_v=np.array([0.3, 0.7, 0.8], dtype=np.float32),
        q_c=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        clv_valid=np.ones(3, dtype=bool),
        n_bins=4,
        shrinkage_strength=10.0,
        degree_bins=2,
    )


def test_inputs_expose_n_and_v_but_not_category_price():
    built = _inputs()
    assert built["user_economic_input"].shape == (3, 5)
    assert built["item_economic_input"].shape == (8, 5)
    np.testing.assert_allclose(
        built["user_economic_input"][:, 0],
        2.0 * built["q_v"] - 1.0,
    )
    np.testing.assert_allclose(
        built["user_activity_gate"], np.array([0.2, 0.6, 0.9])
    )
    assert built["economic_input_diagnostics"]["category_relative_amount_used"] is False
    assert np.isfinite(built["user_mean_economic_position"]).all()


def test_joint_shuffle_moves_q_n_with_the_complete_user_tuple():
    prepared = _inputs()
    shuffled = runner.joint_degree_matched_shuffle(
        prepared, seed=42, degree_bins=2
    )
    for target, source in enumerate(shuffled["source_user"]):
        assert prepared["degree_bin"][target] == prepared["degree_bin"][source]
        assert shuffled["q_n"][target] == prepared["q_n"][source]
        assert (
            shuffled["user_activity_gate"][target]
            == prepared["user_activity_gate"][source]
        )
        np.testing.assert_array_equal(
            shuffled["user_economic_input"][target],
            prepared["user_economic_input"][source],
        )


def test_preflight_describes_explicit_nv_and_test_only_protocol(tmp_path):
    cfg = test_runner.configure_m5_nv_economic_positive_test_run(
        out_dir=str(tmp_path / "results")
    )
    summary = test_runner.preflight_summary(cfg)
    assert summary["validation_constructed"] is False
    assert summary["holdout_evaluation"] is False
    assert summary["m2"]["q_n"] == "post-projection strength gate only"
    assert summary["m2"]["q_c_used_in_m2"] is False
    assert summary["decision"]["interaction_required"] is False


def test_screen_requires_baseline_and_attribution_but_not_interaction():
    metrics = {
        name: {
            "recall@10": 1.0,
            "ndcg@10": 1.0,
            "recall@20": 1.0,
            "ndcg@20": 1.0,
            "recall@50": 1.0,
            "ndcg@50": 1.0,
            "vndcg@10": 1.0,
            "price_purchase_amount_weighted_hit@10": 1.0,
            "coverage@10": 1.0,
            "n_distinct@10": 1.0,
            "top10_share@10": 1.0,
        }
        for name in runner.MODEL_IDS
    }
    metrics[runner.M2_MODEL_ID]["vndcg@10"] = 1.10
    metrics[runner.M4P_MODEL_ID]["vndcg@10"] = 1.10
    metrics[runner.M5_MODEL_ID]["vndcg@10"] = 1.11
    metrics[runner.M5_SHUFFLED_MODEL_ID]["vndcg@10"] = 1.05
    metrics[runner.M5_DEGREE_GATE_MODEL_ID]["vndcg@10"] = 1.04

    reading = runner.screening_reading(metrics)

    assert reading["positive_screen"] is True
    assert reading["primary_interaction_effect_descriptive"] < 0.0
    assert reading["interaction_required"] is False


def test_test_runner_selects_new_screen_without_leaking_global_patch(
    tmp_path, monkeypatch
):
    legacy_screen = test_runner.base.screen
    captured = {}

    def fake_run(cfg):
        captured["screen"] = test_runner.base.screen
        captured["code_version"] = test_runner.base.CODE_VERSION
        return pd.DataFrame([{"ok": True}])

    monkeypatch.setattr(test_runner.base, "run_m5_economic_positive_test", fake_run)
    cfg = test_runner.configure_m5_nv_economic_positive_test_run(
        out_dir=str(tmp_path / "results")
    )

    result = test_runner.run_m5_nv_economic_positive_test(cfg)

    assert bool(result.iloc[0]["ok"])
    assert captured["screen"] is runner
    assert captured["code_version"] == test_runner.CODE_VERSION
    assert test_runner.base.screen is legacy_screen
