from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m3_training_budget as budget


def _cfg(tmp_path, **overrides):
    return budget.configure_m3_budget(
        reference_curve_csv=str(tmp_path / "reference.csv"),
        out_dir=str(tmp_path / "out"), **overrides)


def _reference(path: Path, pref_reg=1e-3):
    pd.DataFrame([
        {"condition": "baseline", "model_id": "m1_bpr_k1", "seed": 42,
         "epoch": epoch, "id_dim": 64, "pref_reg": pref_reg, "rho": 0.0,
         "recall@10": 0.01 + epoch / 1e6, "ndcg@10": 0.02}
        for epoch in range(25, 301, 25)
    ]).to_csv(path, index=False)


def test_preflight_locks_protocol(tmp_path):
    summary = budget.preflight_summary(_cfg(tmp_path))
    assert summary["m3"]["beta"] == 0.15
    assert summary["fixed"]["negative_sampling"] == "one uniform unseen item"
    assert summary["fixed"]["final_test"] is False
    assert summary["fixed"]["holdout"] is False


def test_reference_identity_is_checked(tmp_path):
    cfg = _cfg(tmp_path)
    _reference(Path(cfg.reference_curve_csv))
    assert budget.load_reference_curve(cfg).epoch.tolist() == list(range(25, 301, 25))
    _reference(Path(cfg.reference_curve_csv), pref_reg=1e-4)
    with pytest.raises(ValueError, match="pref_reg"):
        budget.load_reference_curve(cfg)


def test_gap_is_m3_minus_m1():
    m1 = pd.DataFrame({"epoch": [100, 300], "recall@10": [0.01, 0.02]})
    m3 = pd.DataFrame({"epoch": [100, 300], "recall@10": [0.009, 0.021]})
    gap = budget.paired_gap(m1, m3)
    assert gap["recall@10"].tolist() == pytest.approx([-0.001, 0.001])
    assert gap.iloc[1]["recall@10_ratio"] == pytest.approx(1.05)
