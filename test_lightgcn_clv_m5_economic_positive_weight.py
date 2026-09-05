import copy
import json

import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m5_economic_positive_weight as runner


def _train_frame():
    rows = []
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    categories = [0, 0, 1, 1, 0, 0, 1, 1]
    for item, (amount, category) in enumerate(zip(prices, categories)):
        rows.append(
            {
                "u_idx": item % 3,
                "i_idx": item,
                "cat_idx": category,
                "v": amount,
            }
        )
    rows.extend(
        [
            {"u_idx": 0, "i_idx": 0, "cat_idx": 0, "v": 2.0},
            {"u_idx": 1, "i_idx": 7, "cat_idx": 1, "v": 16.0},
        ]
    )
    return pd.DataFrame(rows)


def _economic_inputs():
    return runner.build_economic_inputs(
        _train_frame(),
        n_users=3,
        n_items=8,
        q_v=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        q_c=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        clv_valid=np.ones(3, dtype=bool),
        n_bins=4,
        shrinkage_strength=10.0,
        degree_bins=2,
    )


def test_economic_inputs_use_equal_item_bins_and_preserve_shrinkage():
    built = _economic_inputs()

    np.testing.assert_array_equal(
        np.bincount(built["item_bin"], minlength=4), np.array([2, 2, 2, 2])
    )
    assert built["user_economic_input"].shape == (3, 5)
    assert built["item_economic_input"].shape == (8, 2)
    assert np.all(np.linalg.norm(built["user_economic_input"][:, :4], axis=1) < 1)
    assert built["economic_input_diagnostics"]["item_count_bin_imbalance"] == 0


def test_joint_shuffle_moves_complete_tuple_inside_degree_bin():
    prepared = _economic_inputs()
    shuffled = runner.joint_degree_matched_shuffle(
        prepared, seed=42, degree_bins=2
    )

    for target, source in enumerate(shuffled["source_user"]):
        assert prepared["degree_bin"][target] == prepared["degree_bin"][source]
        np.testing.assert_array_equal(
            shuffled["user_economic_input"][target],
            prepared["user_economic_input"][source],
        )
        assert shuffled["q_c"][target] == prepared["q_c"][source]


def _config(tmp_path):
    return runner.configure_m5_economic_positive_run(
        out_dir=str(tmp_path / "results"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )


def test_preflight_freezes_six_arms_and_protected_splits(tmp_path):
    summary = runner.preflight_summary(_config(tmp_path))

    assert summary["reported_models"] == list(runner.MODEL_IDS)
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["m2"]["rho"] == 0.15
    assert summary["m4_prime"]["lambda"] == 0.5
    assert summary["reading_rule"]["primary_metric"] == "vndcg@10"


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.05},
        {"positive_weight_lambda": 0.2},
        {"economic_bins": 3},
        {"shrinkage_strength": 0.0},
        {"negative_count": 1},
        {"n_layers": 1},
    ],
)
def test_fast_screen_rejects_unplanned_changes(tmp_path, override):
    with pytest.raises(ValueError, match="M5 screen"):
        runner.configure_m5_economic_positive_run(
            out_dir=str(tmp_path / "results"),
            baseline_result_dir=str(tmp_path / "baseline"),
            **override,
        )


def _metrics(value=1.0):
    return {
        "recall@10": value,
        "ndcg@10": value,
        "recall@20": value,
        "ndcg@20": value,
        "recall@50": value,
        "ndcg@50": value,
        "vndcg@10": value,
        "coverage@10": value,
        "n_distinct@10": value,
        "top10_share@10": value,
    }


def _passing_rows():
    rows = {
        runner.M1_MODEL_ID: _metrics(1.0),
        runner.M2_MODEL_ID: _metrics(1.01),
        runner.M4P_MODEL_ID: _metrics(1.02),
        runner.M5_MODEL_ID: _metrics(1.06),
        runner.M5_SHUFFLED_MODEL_ID: _metrics(1.04),
        runner.M5_DEGREE_GATE_MODEL_ID: _metrics(1.03),
    }
    for metrics in rows.values():
        metrics["coverage@10"] = 1.0
        metrics["n_distinct@10"] = 1.0
        metrics["top10_share@10"] = 1.0
    return rows


def test_interaction_is_factorial_difference_in_differences():
    rows = runner.interaction_rows(_passing_rows()).set_index("metric")

    assert rows.at["vndcg@10", "interaction_effect"] == pytest.approx(0.03)


def test_screen_requires_primary_attribution_interaction_accuracy_and_exposure():
    passing = _passing_rows()
    assert runner.screening_reading(passing)["positive_screen"] is True

    shuffled_tie = copy.deepcopy(passing)
    shuffled_tie[runner.M5_SHUFFLED_MODEL_ID]["vndcg@10"] = 1.06
    assert runner.screening_reading(shuffled_tie)["positive_screen"] is False

    accuracy_loss = copy.deepcopy(passing)
    accuracy_loss[runner.M5_MODEL_ID]["recall@10"] = 0.98
    assert runner.screening_reading(accuracy_loss)["positive_screen"] is False


def test_colab_is_valid_json_and_calls_one_development_screen():
    path = runner.Path("clv_m5_economic_positive_weight_dunnhumby_colab.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )

    assert "run_m5_economic_positive_screen" in source
    assert "REVIEWED_SHA" in source
    assert "EVAL_HOLDOUT=True" not in source.replace(" ", "")
