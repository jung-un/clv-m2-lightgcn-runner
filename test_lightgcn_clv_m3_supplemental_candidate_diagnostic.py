from types import SimpleNamespace

import numpy as np
import torch

from clv_m3_clv_conditioned_candidate_item_graph import (
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
)
from lightgcn_clv_m3_supplemental_candidate_diagnostic import (
    candidate_truth_set_reading,
)


def _operator(pairs):
    indices = torch.tensor(pairs, dtype=torch.long).T
    values = torch.ones(len(pairs), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (2, 8)).coalesce()


def _tiny_graph():
    base = _operator([(0, 1), (1, 1)])
    extras = {
        ARM_GENERAL: _operator([(0, 5)]),
        ARM_ACTUAL: _operator([(0, 4), (0, 5), (1, 6)]),
        ARM_SHUFFLE: _operator([(0, 4), (1, 7)]),
    }
    full = {}
    for arm, extra in extras.items():
        indices = torch.cat([base.indices(), extra.indices()], dim=1)
        values = torch.ones(indices.shape[1], dtype=torch.float32)
        full[arm] = torch.sparse_coo_tensor(
            indices, values, (2, 8)
        ).coalesce()
    return SimpleNamespace(
        user_item_operators=full,
        candidate_blocks={"base": base, **extras},
        clv_percentile=np.array([0.25, 0.75]),
    )


def test_candidate_truth_set_reading_separates_net_from_actual_only():
    frame, reading = candidate_truth_set_reading(
        graph=_tiny_graph(),
        test_pairs={(0, 4), (0, 5), (1, 6), (1, 7)},
    )

    assert reading["actual_truth_hits"] - reading["shuffle_truth_hits"] == 1
    assert reading["actual_extra_truth_hits"] == 3
    assert reading["shuffle_extra_truth_hits"] == 2
    assert reading["actual_only_truth_pairs"] == 2
    assert reading["actual_only_inside_general_truth_pairs"] == 1
    assert reading["actual_only_outside_general_truth_pairs"] == 1
    assert reading["automatic_model_selection"] is False
    assert reading["precheck_passed"] is True
    assert set(frame["clv_group"]) == {"Q2", "Q4", "전체"}


def test_candidate_truth_set_reading_accepts_ground_truth_dictionary():
    _, reading = candidate_truth_set_reading(
        graph=_tiny_graph(),
        test_pairs={
            0: np.array([4, 5], dtype=np.int32),
            1: np.array([6, 7], dtype=np.int32),
        },
    )

    assert reading["truth_pairs"] == 4
    assert reading["actual_only_outside_general_truth_pairs"] == 1


def test_candidate_truth_set_reading_rejects_missing_blocks():
    graph = _tiny_graph()
    graph.candidate_blocks = None

    try:
        candidate_truth_set_reading(graph, {(0, 4)})
    except ValueError as error:
        assert "candidate blocks" in str(error)
    else:
        raise AssertionError("missing candidate blocks must be rejected")
