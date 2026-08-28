import json
from pathlib import Path

import numpy as np
import pytest
import torch

import lightgcn_clv_dynamic_multianchor_diagnostic as diagnostic


def test_topk_change_table_separates_training_path_from_direct_clv_effect():
    users = np.array([0, 1], dtype=np.int64)
    segments = np.array(["저CLV", "고CLV"], dtype=object)
    m1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    m2_rho0 = np.array([[1, 2, 3], [4, 7, 6]], dtype=np.int64)
    m2 = np.array([[1, 8, 3], [4, 7, 6]], dtype=np.int64)

    frame = diagnostic.topk_change_table(
        users=users,
        segments=segments,
        topk_by_view={"m1": m1, "m2_rho0": m2_rho0, "m2": m2},
        ks=(2,),
    )

    overall = frame[frame["segment"].eq("전체")].set_index("comparison")
    assert overall.at["m1_vs_m2_total", "changed_user_share"] == 1.0
    assert overall.at["m1_vs_m2_rho0_joint_training", "changed_user_share"] == 0.5
    assert overall.at["m2_rho0_vs_m2_direct_clv", "changed_user_share"] == 0.5
    assert overall.at["m1_vs_m2_total", "mean_jaccard"] == pytest.approx(1 / 3)


def test_condition_variation_table_excludes_invalid_users_and_reports_change():
    conditions = np.array(
        [
            [0.0, 0.0, 9.0],
            [0.0, 0.1, 9.0],
            [0.0, 0.2, 9.0],
        ],
        dtype=np.float64,
    )
    valid = np.array([True, True, False])

    frame = diagnostic.condition_variation_table(
        conditions,
        valid=valid,
        scope="training_anchors",
    )
    row = frame.iloc[0]

    assert row["valid_user_count"] == 2
    assert row["unchanged_user_share"] == 0.5
    assert row["changed_gt_0_05_user_share"] == 0.5
    assert row["mean_abs_first_last_change"] == pytest.approx(0.1)


def test_score_component_statistics_preserves_exact_delta_decomposition():
    m1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m2_rho0 = torch.tensor([1.1, 1.9, 3.0, 4.0])
    m2 = torch.tensor([1.2, 1.8, 3.1, 3.9])

    frame = diagnostic.score_component_statistics(m1, m2_rho0, m2)
    indexed = frame.set_index("component")

    assert indexed.at["joint_training_path", "mean"] == pytest.approx(0.0)
    assert indexed.at["direct_clv_condition", "std"] == pytest.approx(0.1)
    assert indexed.at["total_m2_minus_m1", "std"] == pytest.approx(
        np.sqrt(0.025)
    )
    assert indexed.at["total_m2_minus_m1", "max_decomposition_error"] < 1e-6


def test_rank_boundary_table_reports_actual_and_potential_crossing():
    frame = diagnostic.rank_boundary_table(
        ks=(10,),
        margins={10: np.array([0.1, 0.2])},
        direct_shift_ranges=np.array([0.05, 0.4]),
        direct_changed={10: np.array([False, True])},
        total_changed={10: np.array([True, True])},
    )
    row = frame.iloc[0]

    assert row["direct_topk_changed_user_share"] == 0.5
    assert row["total_topk_changed_user_share"] == 1.0
    assert row["shift_range_ge_margin_user_share"] == 0.5
    assert row["median_shift_range_to_margin"] == pytest.approx(1.25)


def test_colab_runs_checkpoint_diagnostic_without_training():
    notebook = json.loads(
        Path(
            "clv_m2_dynamic_clv_multianchor_checkpoint_diagnostic_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "run_dynamic_multianchor_diagnostic" in source
    assert "run_dynamic_multianchor(" not in source
    assert "summary['training'] is False" in source
