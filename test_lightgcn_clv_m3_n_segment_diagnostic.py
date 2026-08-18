from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m3_n_segment_diagnostic as diagnostic


def test_segment_decision_requires_both_n_oriented_metrics_positive():
    frame = pd.DataFrame(
        [
            {
                "segment": "N-oriented (pi_N>0.5)",
                "revenue@10_delta": 0.01,
                "recall@20_delta": 0.001,
            }
        ]
    )
    assert diagnostic.segment_decision(frame)["proceed_to_compositional_m3"] is True

    frame.loc[0, "recall@20_delta"] = -0.001
    assert diagnostic.segment_decision(frame)["proceed_to_compositional_m3"] is False


def test_missing_checkpoint_fails_without_training(tmp_path):
    missing = Path(tmp_path) / "missing.pt"
    with pytest.raises(FileNotFoundError, match="never trains"):
        diagnostic._require_existing_checkpoint(missing, "M3-N")
