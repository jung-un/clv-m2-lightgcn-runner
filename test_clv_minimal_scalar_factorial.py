import numpy as np
import pandas as pd
import pytest
import torch

from clv_m3_direct_value_graph import build_direct_clv_value_graph
import lightgcn_clv_minimal_scalar_factorial as screen
import lightgcn_clv_minimal_scalar_factorial_test as runner
import lightgcn_clv_v3 as v3


def _train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2],
            "i_idx": [0, 1, 0, 2, 2],
            "v": [2.0, 3.0, 1.0, 4.0, 5.0],
            "up": [2.0, 3.0, 1.0, 4.0, 5.0],
        }
    )


def _inputs():
    return screen.build_minimal_scalar_clv_inputs(
        _train(),
        n_users=3,
        n_items=3,
        q_c=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        clv_valid=np.ones(3, dtype=bool),
    )


def test_scalar_inputs_use_only_q_c_and_item_purchaser_context():
    built = _inputs()

    np.testing.assert_allclose(
        built["user_economic_input"][:, 0], [-0.8, 0.0, 0.8]
    )
    np.testing.assert_allclose(
        built["item_economic_input"][:, 0], [-0.4, -0.8, 0.4]
    )
    np.testing.assert_array_equal(built["item_amount_percentile"], np.ones(3))
    diagnostics = built["economic_input_diagnostics"]
    assert diagnostics["historical_clv_proxy"] == "q_C = percentile(n_u * v_u)"
    assert diagnostics["item_context_is_item_clv"] is False
    assert diagnostics["m4_item_feature"] == "constant_one_no_item_price"


def test_m3_uses_same_edges_and_only_user_q_c_weights():
    train = _train()
    q_c = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    graph = build_direct_clv_value_graph(
        train, n_users=3, n_items=3, clv_gate=q_c, alpha=screen.M3_ALPHA
    )

    keys = np.unique(
        train["u_idx"].to_numpy(np.int64) * 3
        + train["i_idx"].to_numpy(np.int64)
    )
    np.testing.assert_array_equal(graph.edge_users, keys // 3)
    np.testing.assert_array_equal(graph.edge_items, keys % 3)
    for user in range(3):
        weights = graph.user_clv_weights[graph.edge_users == user]
        assert np.unique(weights).size == 1
    assert graph.user_clv_weights[graph.edge_users == 2][0] > graph.user_clv_weights[
        graph.edge_users == 0
    ][0]


def test_m3_weighted_adjacency_differs_from_binary_but_keeps_shape():
    train = _train()
    graph = build_direct_clv_value_graph(
        train,
        n_users=3,
        n_items=3,
        clv_gate=np.array([0.1, 0.5, 0.9]),
        alpha=screen.M3_ALPHA,
    )
    binary = v3.build_adj(
        graph.edge_users, graph.edge_items, np.ones(len(graph.edge_users)), 3, 3
    ).cpu()
    weighted = v3.build_adj(
        graph.edge_users, graph.edge_items, graph.user_clv_weights, 3, 3
    ).cpu()

    assert binary.shape == weighted.shape == (6, 6)
    assert not torch.equal(binary.values(), weighted.values())


def test_arm_specifications_are_exactly_m1_to_m5():
    prepared = {"placeholder": True}
    cfg = runner.configure_minimal_scalar_clv_test_run(out_dir="/tmp/scalar-clv")
    run_cfg = runner.base._screen_config(cfg, 42)

    specs = screen.arm_specifications(prepared, run_cfg)

    assert [spec["model_id"] for spec in specs] == list(screen.MODEL_IDS)
    assert [(spec["rho"], spec["m3"], spec["weighted"]) for spec in specs] == [
        (0.0, False, False),
        (0.15, False, False),
        (0.0, True, False),
        (0.0, False, True),
        (0.15, True, True),
    ]


def test_constant_item_feature_makes_m4_weight_user_clv_only():
    prepared = {
        "data": {"tr_u": np.array([0, 1, 1]), "tr_i": np.array([0, 0, 1])},
        "item_amount_percentile": np.ones(2, dtype=np.float32),
    }
    assignment = {"q_c": np.array([0.2, 0.8], dtype=np.float32)}
    normalizer = screen.legacy._train_weight_normalizer(prepared, assignment, 0.5)

    expected = np.mean([1.1, 1.4, 1.4])
    assert normalizer == pytest.approx(expected)


def test_seed42_config_locks_minimal_run(tmp_path):
    cfg = runner.configure_minimal_scalar_clv_test_run(
        out_dir=str(tmp_path / "results")
    )

    assert cfg.seeds == (42,)
    assert cfg.negative_count == 1
    assert cfg.economic_dim == 1
    assert cfg.rho == 0.15
    assert cfg.positive_weight_lambda == 0.5
    summary = runner.preflight_summary(cfg)
    assert len(summary["models"]) == 5
    assert summary["fixed"]["new_item_task"] is True
    assert "price or price-bin inputs" in summary["excluded"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", tuple(range(42, 52))),
        ("negative_count", 5),
        ("economic_dim", 4),
        ("epochs", 99),
        ("n_layers", 3),
        ("rho", 0.25),
    ],
)
def test_seed42_config_rejects_unrequested_changes(tmp_path, field, value):
    with pytest.raises(ValueError):
        runner.configure_minimal_scalar_clv_test_run(
            out_dir=str(tmp_path / "results"), **{field: value}
        )
