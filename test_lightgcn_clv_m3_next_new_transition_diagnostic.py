import json

import numpy as np
import pandas as pd
import pytest

from lightgcn_clv_m3_next_new_transition_diagnostic import (
    M3NextNewTransitionDiagnosticConfig,
    _build_truth,
    _prepare_transactions_frame,
    configure_m3_next_new_transition_diagnostic,
    preflight_summary,
    run_m3_next_new_transition_diagnostic_from_frames,
    validate_config,
)


def _raw_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (10, 100, 660, "a", 4.0, 4.0),
            (10, 101, 662, "b", 6.0, 6.0),
            (20, 100, 661, "c", 5.0, 5.0),
            (20, 102, 662, "d", 8.0, 8.0),
            # pseudo-future: 102 is new for user 10; 100 is repeat for user 20
            (10, 102, 663, "e", 7.0, 7.0),
            (20, 100, 664, "f", 5.0, 5.0),
            # must be discarded before any feature or universe is constructed
            (10, 999, 670, "future", 999.0, 999.0),
        ],
        columns=["u_raw", "i_raw", "t", "b_raw", "v", "up"],
    )


def _cfg(tmp_path) -> M3NextNewTransitionDiagnosticConfig:
    return configure_m3_next_new_transition_diagnostic(out_dir=str(tmp_path))


def test_config_is_locked_to_train_only_historical_interval(tmp_path):
    cfg = _cfg(tmp_path)
    summary = preflight_summary(cfg)
    assert summary["construction_interval"]["end_inclusive"] == 662
    assert summary["pseudo_future_interval"] == {
        "start_inclusive": 663,
        "end_inclusive": 669,
    }
    assert summary["final_test_constructed"] is False
    assert summary["holdout_constructed"] is False
    assert summary["min_item_interactions"] == 1

    with pytest.raises(ValueError):
        validate_config(
            M3NextNewTransitionDiagnosticConfig(
                construction_end_day=663, out_dir=str(tmp_path)
            )
        )


def test_prepare_caps_input_before_indexing_and_feature_construction(tmp_path):
    prepared = _prepare_transactions_frame(_raw_transactions(), _cfg(tmp_path))
    assert prepared["transactions"]["t"].max() == 664
    assert prepared["construction"]["t"].max() == 662
    assert 999 not in set(prepared["item_ids"])
    assert set(prepared["transactions"]["basket_id"]) == {"a", "b", "c", "d", "e", "f"}


def test_truth_excludes_construction_pairs_and_uses_only_pseudo_future(tmp_path):
    prepared = _prepare_transactions_frame(_raw_transactions(), _cfg(tmp_path))
    truth, truth_value = _build_truth(prepared)

    user10 = int(np.flatnonzero(prepared["user_ids"] == 10)[0])
    item102 = int(np.flatnonzero(prepared["item_ids"] == 102)[0])
    assert set(truth) == {user10}
    np.testing.assert_array_equal(truth[user10], [item102])
    np.testing.assert_allclose(truth_value[user10], [7.0])


def test_end_to_end_frame_run_writes_integrity_checked_artifacts(tmp_path):
    result = run_m3_next_new_transition_diagnostic_from_frames(
        _raw_transactions(),
        _cfg(tmp_path),
        input_manifest={"synthetic": True},
        source_revision="test-revision",
    )

    assert result.attrs["quality_passed"] is True
    assert set(result.model_id) == {
        "transition_global",
        "transition_clv",
        "transition_clv_shuffle",
    }
    paths = result.attrs["result_paths"]
    for path in paths.values():
        assert tmp_path.joinpath(path.split("/")[-1]).exists()
    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    assert payload["source_revision"] == "test-revision"
    assert payload["quality_passed"] is True
    assert payload["split"]["transactions_after_day_669"] == 0
