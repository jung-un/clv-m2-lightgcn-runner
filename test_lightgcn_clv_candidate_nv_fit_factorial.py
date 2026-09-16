import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from clv_candidate_nv_fit_model import (
    CandidateNVFitLightGCN,
    candidate_nv_fit_numpy,
)
import lightgcn_clv_candidate_nv_fit_factorial as screen
import lightgcn_clv_v3 as v3


def _train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2],
            "i_idx": [0, 0, 1, 0, 2, 2],
            "cat_idx": [0, 0, 0, 0, 1, 1],
            "v": [2.0, 3.0, 4.0, 2.0, 8.0, 6.0],
            "up": [2.0, 3.0, 4.0, 2.0, 8.0, 6.0],
        }
    )


def _candidate_inputs():
    return screen.build_candidate_nv_inputs(
        _train(),
        n_users=3,
        n_items=3,
        q_n=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        q_v=np.array([0.3, 0.5, 0.8], dtype=np.float32),
        q_c=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        clv_valid=np.ones(3, dtype=bool),
    )


def _adj():
    return v3.build_adj(
        np.array([0, 0, 1, 1, 2]),
        np.array([0, 1, 0, 2, 2]),
        np.ones(5, dtype=np.float32),
        3,
        3,
    ).cpu()


def _model(rho=0.15):
    built = _candidate_inputs()
    torch.manual_seed(7)
    return CandidateNVFitLightGCN(
        n_users=3,
        n_items=3,
        user_q_n=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        user_q_v=np.array([0.3, 0.5, 0.8], dtype=np.float32),
        user_q_c=np.array([0.1, 0.5, 0.9], dtype=np.float32),
        user_clv_valid=np.ones(3, dtype=bool),
        item_buyer_q_n=built["item_buyer_q_n"],
        item_amount_percentile=built["item_amount_percentile"],
        item_context_valid=built["item_context_valid"],
        adj=_adj(),
        id_dim=4,
        rho=rho,
        n_layers=2,
    )


def test_candidate_fit_is_pair_specific_and_bounded():
    fit = candidate_nv_fit_numpy(
        np.array([0.2, 0.2]),
        np.array([0.8, 0.8]),
        np.array([0.2, 0.9]),
        np.array([0.8, 0.1]),
    )

    np.testing.assert_allclose(fit, [1.0, 0.3])
    assert np.all((fit >= 0.0) & (fit <= 1.0))


def test_item_buyer_activity_context_uses_distinct_user_item_pairs():
    built = _candidate_inputs()

    # Item 0 has buyers u0 and u1.  The repeated u0 row must not count twice.
    assert built["item_buyer_q_n"][0] == pytest.approx(0.4)
    assert built["item_buyer_q_n"][1] == pytest.approx(0.2)
    assert built["item_buyer_q_n"][2] == pytest.approx(0.75)
    assert built["economic_input_diagnostics"]["item_context_is_item_clv"] is False


def test_m3_keeps_edge_set_and_each_user_total_mass():
    built = _candidate_inputs()
    pos_key = np.array([0, 1, 3, 5, 8], dtype=np.int64)
    prepared = {
        **built,
        "data": {"n_users": 3, "n_items": 3, "pos_key": pos_key},
    }

    adjacency, diagnostics = screen.build_candidate_m3_adjacency(
        prepared, beta=0.15
    )

    assert adjacency.shape == (6, 6)
    assert diagnostics["edge_count"] == len(pos_key)
    assert diagnostics["user_mass_preservation_max_abs_error"] < 1e-6
    assert diagnostics["coefficient_min"] > 0.0
    assert diagnostics["changed_edge_share"] > 0.0


def test_axis_weights_sum_to_two_and_rho_zero_is_plain_lightgcn():
    model = _model(rho=0.0)
    user, item = model.propagated_embeddings()
    user_id, item_id = model.id_embeddings()

    torch.testing.assert_close(model.axis_weights().sum(), torch.tensor(2.0))
    torch.testing.assert_close(user @ item.T, user_id @ item_id.T)
    assert model.representation_diagnostics()["external_reranking"] is False


def test_six_arms_isolate_m2_m3_m4_and_two_combinations():
    cfg = screen.configure_candidate_nv_fit_screen(
        out_dir="/tmp/candidate-nv", baseline_result_dir="/tmp/base"
    )
    specs = screen.arm_specifications({"placeholder": True}, cfg)

    assert [spec["model_id"] for spec in specs] == list(screen.MODEL_IDS)
    assert [(spec["m2"], spec["m3"], spec["weighted"]) for spec in specs] == [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (True, True, True),
    ]


def test_config_locks_development_split_and_single_negative():
    cfg = screen.configure_candidate_nv_fit_screen(
        out_dir="/tmp/candidate-nv", baseline_result_dir="/tmp/base"
    )
    summary = screen.preflight_summary(cfg)

    assert cfg.negative_count == 1
    assert cfg.rho == 0.15
    assert cfg.beta_m3 == 0.15
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["item_context_not_item_clv"]["b_n"].startswith("mean q_N")
    with pytest.raises(ValueError, match="negative_count"):
        screen.configure_candidate_nv_fit_screen(
            out_dir="/tmp/candidate-nv",
            baseline_result_dir="/tmp/base",
            negative_count=5,
        )


def _metrics(value):
    return {
        "recall@10": value,
        "ndcg@10": value,
        "price_purchase_amount_weighted_hit@10": value,
        "vndcg@10": value,
    }


def test_reading_separates_baseline_direction_and_combination_increment():
    rows = {
        screen.M1_MODEL_ID: _metrics(1.0),
        screen.M2_MODEL_ID: _metrics(1.01),
        screen.M3_MODEL_ID: _metrics(0.99),
        screen.M4_MODEL_ID: _metrics(1.02),
        screen.M5A_MODEL_ID: _metrics(1.03),
        screen.M5B_MODEL_ID: _metrics(1.025),
    }

    reading = screen.screening_reading(rows)

    assert reading["classification"] == "combination_baseline_direction"
    assert reading["m5a_beats_m1_on_all_four_top10_metrics"] is True
    assert reading["m5a_beats_m4_on_all_four_top10_metrics"] is True
    assert reading["clv_attribution_tested"] is False


def test_result_metadata_keeps_wide_dataframe_repr_safe():
    frame = pd.DataFrame(np.zeros((6, 207)))
    comparison = pd.DataFrame({"metric": ["recall@10"]})
    overlap = pd.DataFrame({"changed_user_share": [0.1]})
    score_diagnostics = pd.DataFrame({"model_id": [screen.M2_MODEL_ID]})

    screen.attach_result_metadata(
        frame,
        comparison=comparison,
        overlap=overlap,
        score_diagnostics=score_diagnostics,
        mechanism={"candidate_fit": {"std": 0.1}},
        reading={"classification": "directional_nonpass"},
        paths={"json": Path("/tmp/result.json")},
    )

    assert frame.attrs["comparison"] == comparison.to_dict("records")
    assert frame.attrs["top10_overlap"] == overlap.to_dict("records")
    assert frame.attrs["score_diagnostics"] == score_diagnostics.to_dict("records")
    assert "[6 rows x 207 columns]" in repr(frame)


def test_colab_pins_reviewed_source_and_runs_six_arm_screen_once():
    path = Path("clv_m5_candidate_nv_fit_factorial_dunnhumby_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = "\n".join(code)

    for cell in code:
        if not cell.lstrip().startswith(("%", "!")):
            # IPython magics only occur as standalone lines in the setup cell.
            cleaned = "\n".join(
                line for line in cell.splitlines() if not line.startswith("%")
            )
            ast.parse(cleaned)
    assert "64c55158ed506549a763895861a2cfe4b1c353d2" in source
    assert source.count("result_df = screen.run_candidate_nv_fit_screen(cfg)") == 1
    assert "cfg.negative_count == 1" in source
    assert "len(summary['trained_models']) == 6" in source
    assert "final_test_constructed" in source
    assert "holdout_constructed" in source
    assert "pd.DataFrame(frame)" in source
