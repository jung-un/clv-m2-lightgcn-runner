"""Train-free pair-set diagnostic for the supplemental M3 relation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clv_m3_clv_conditioned_candidate_item_graph import (
    ACTIVE_ARMS,
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
    RELATION_MODE_SUPPLEMENTAL,
)
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_m3_clv_conditioned_candidate_item as runner


CODE_VERSION = "m3-clv-supplemental-candidate-set-precheck-v1"


def _operator_pairs(operator) -> set[tuple[int, int]]:
    users, items = operator.coalesce().indices().cpu().numpy()
    return {
        (int(user), int(item))
        for user, item in zip(users, items, strict=True)
    }


def _truth_pairs(
    test_pairs: dict[int, np.ndarray] | set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if isinstance(test_pairs, dict):
        return {
            (int(user), int(item))
            for user, items in test_pairs.items()
            for item in np.asarray(items, dtype=np.int64)
        }
    return {(int(user), int(item)) for user, item in test_pairs}


def _quintile_labels(q_values: np.ndarray) -> np.ndarray:
    labels = np.floor(np.asarray(q_values, dtype=np.float64) * 5).astype(int)
    labels = np.clip(labels, 0, 4) + 1
    return np.asarray([f"Q{value}" for value in labels], dtype=object)


def candidate_truth_set_reading(
    graph,
    test_pairs: dict[int, np.ndarray] | set[tuple[int, int]],
) -> tuple[pd.DataFrame, dict]:
    """Separate net hit counts from actual-only and shuffle-only truth pairs."""
    if not graph.candidate_blocks:
        raise ValueError("supplemental candidate blocks are required")
    required = {"base", ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE}
    missing = required - set(graph.candidate_blocks)
    if missing:
        raise ValueError(f"candidate blocks are missing {sorted(missing)}")

    truth = _truth_pairs(test_pairs)
    base = _operator_pairs(graph.candidate_blocks["base"])
    extra = {
        arm: _operator_pairs(graph.candidate_blocks[arm])
        for arm in ACTIVE_ARMS
    }
    full = {arm: base | extra[arm] for arm in ACTIVE_ARMS}
    full_hits = {arm: full[arm] & truth for arm in ACTIVE_ARMS}
    extra_hits = {arm: extra[arm] & truth for arm in ACTIVE_ARMS}

    actual_only = extra_hits[ARM_ACTUAL] - extra_hits[ARM_SHUFFLE]
    shuffle_only = extra_hits[ARM_SHUFFLE] - extra_hits[ARM_ACTUAL]
    actual_only_inside_general = actual_only & full_hits[ARM_GENERAL]
    actual_only_outside_general = actual_only - full_hits[ARM_GENERAL]

    q_labels = _quintile_labels(graph.clv_percentile)
    truth_users = {user for user, _ in truth}
    populations = [
        (
            label,
            {user for user in truth_users if q_labels[user] == label},
        )
        for label in sorted(set(q_labels[list(truth_users)]))
    ]
    populations.append(("전체", truth_users))
    rows = []
    for label, users in populations:
        group_truth = {pair for pair in truth if pair[0] in users}
        for block_name, candidates in (
            ("full", full),
            ("supplemental", extra),
        ):
            for arm in ACTIVE_ARMS:
                group_candidates = {
                    pair for pair in candidates[arm] if pair[0] in users
                }
                hits = group_candidates & group_truth
                rows.append(
                    {
                        "clv_group": label,
                        "candidate_block": block_name,
                        "graph_arm": arm,
                        "n_users": int(len(users)),
                        "truth_pairs": int(len(group_truth)),
                        "candidate_pairs": int(len(group_candidates)),
                        "candidate_truth_hits": int(len(hits)),
                        "candidate_truth_pair_coverage": float(
                            len(hits) / len(group_truth)
                            if group_truth
                            else 0.0
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    precheck_passed = bool(
        len(extra_hits[ARM_ACTUAL]) > len(extra_hits[ARM_SHUFFLE])
        and len(actual_only_outside_general) > 0
    )
    reading = {
        "automatic_model_selection": False,
        "analysis_type": "descriptive post-hoc pair-set precheck",
        "truth_pairs": int(len(truth)),
        "general_truth_hits": int(len(full_hits[ARM_GENERAL])),
        "actual_truth_hits": int(len(full_hits[ARM_ACTUAL])),
        "shuffle_truth_hits": int(len(full_hits[ARM_SHUFFLE])),
        "general_extra_truth_hits": int(len(extra_hits[ARM_GENERAL])),
        "actual_extra_truth_hits": int(len(extra_hits[ARM_ACTUAL])),
        "shuffle_extra_truth_hits": int(len(extra_hits[ARM_SHUFFLE])),
        "actual_only_truth_pairs": int(len(actual_only)),
        "shuffle_only_truth_pairs": int(len(shuffle_only)),
        "actual_only_inside_general_truth_pairs": int(
            len(actual_only_inside_general)
        ),
        "actual_only_outside_general_truth_pairs": int(
            len(actual_only_outside_general)
        ),
        "precheck_passed": precheck_passed,
        "routing": (
            "The previously observed CLV candidate signal survives outside "
            "the general relation; performance evaluation may proceed only "
            "on an approved new test interval or independent data."
            if precheck_passed
            else "The CLV extra-candidate signal does not survive outside the "
            "general relation; stop M3 structural search without training."
        ),
        "limitation": (
            "DAY 684--690 was already inspected; this diagnostic is not a new "
            "performance test and cannot support significance or generalization."
        ),
    }
    return frame, reading


def run_supplemental_candidate_precheck(
    cfg: runner.CLVCandidateItemConfig | None = None,
) -> pd.DataFrame:
    """Build train-only relations and inspect old truth sets without training."""
    cfg = cfg or runner.configure_clv_candidate_item_supplemental_run()
    cfg = runner.validate_clv_candidate_item_config(cfg)
    if cfg.relation_mode != RELATION_MODE_SUPPLEMENTAL:
        raise ValueError("supplemental precheck requires its fixed relation mode")
    if cfg.evaluation_authorized or cfg.time_cutoff != 690:
        raise ValueError(
            "the train-free precheck is fixed to the already-seen DAY 684--690"
        )

    preflight = runner.preflight_summary(cfg)
    prepared = runner._prepare(cfg)
    frame, reading = candidate_truth_set_reading(
        prepared["graph"], prepared["data"]["splits"]["test"][0]
    )
    stem = f"m3_clv_supplemental_candidate_precheck_{prepared['config_hash']}"
    paths = {
        "set_csv": Path(cfg.out_dir) / f"{stem}.csv",
        "json": Path(cfg.out_dir) / f"{stem}.json",
    }
    fixed_train._atomic_csv(paths["set_csv"], frame)
    payload = {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc pair-set precheck",
        "training": False,
        "checkpoint_selection": False,
        "final_test_constructed": False,
        "holdout_constructed": False,
        "source_split": "DAY 1--683 train; DAY 684--690 inspected truth",
        "config": asdict(cfg),
        "preflight": preflight,
        "graph_diagnostics": prepared["graph"].diagnostics,
        "set_rows": frame.to_dict("records"),
        "reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    fixed_train._atomic_json(paths["json"], payload)
    frame.attrs["reading"] = reading
    frame.attrs["graph_diagnostics"] = prepared["graph"].diagnostics
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


__all__ = [
    "CODE_VERSION",
    "candidate_truth_set_reading",
    "run_supplemental_candidate_precheck",
]
