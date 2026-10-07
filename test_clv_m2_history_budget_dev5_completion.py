import json
from pathlib import Path

import pandas as pd
import pytest

import clv_m2_history_budget_dev5_completion as completion


METRICS = {
    "recall@10": (0.100, 0.101),
    "ndcg@10": (0.200, 0.199),
    "recall@50": (0.300, 0.306),
    "price_purchase_amount_weighted_hit@50": (1.000, 1.030),
}


def _curve(seed: int, *, favorable: bool = True) -> pd.DataFrame:
    rows = []
    for model_id in (completion.M1_MODEL_ID, completion.M2_MODEL_ID):
        row = {
            "condition": "baseline",
            "model_id": model_id,
            "seed": seed,
            "epoch": 300,
            "id_dim": 64,
            "axis_dim": 0 if model_id == completion.M1_MODEL_ID else 4,
            "pref_reg": 0.001,
            "rho": 0.0 if model_id == completion.M1_MODEL_ID else 0.05,
        }
        for metric, (m1, m2) in METRICS.items():
            value = m1 if model_id == completion.M1_MODEL_ID else m2
            if not favorable and model_id == completion.M2_MODEL_ID and metric == "recall@50":
                value = m1 - 0.01
            row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_completion_contract_only_runs_missing_development_seeds(tmp_path):
    cfg = completion.configure(out_dir=str(tmp_path))

    summary = completion.preflight_summary(cfg)

    assert cfg.seeds == (45, 46)
    assert cfg.epochs == 300
    assert cfg.eval_every == 25
    assert cfg.conditions == ("baseline",)
    assert summary["split"] == "historical_development_days_684_690"
    assert summary["final_test_constructed"] is False
    assert summary["holdout_constructed"] is False
    assert summary["new_training_fits"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seeds", (45,)),
        ("seeds", (45, 46, 47)),
        ("epochs", 100),
        ("eval_every", 100),
        ("conditions", ("baseline", "wide")),
    ],
)
def test_completion_rejects_protocol_drift(tmp_path, field, value):
    cfg = completion.configure(out_dir=str(tmp_path))
    with pytest.raises(ValueError):
        completion.validate_config(cfg.__class__(**({**cfg.__dict__, field: value})))


def test_five_seed_reading_applies_pre_registered_role_gate():
    curve = pd.concat([_curve(seed) for seed in range(42, 47)], ignore_index=True)

    absolute, paired, summary, reading = completion.read_five_seed_result(curve)

    assert len(absolute) == 10
    assert len(paired) == 5
    assert set(summary.metric) == set(METRICS)
    assert reading["recall50_mean_positive"] is True
    assert reading["recall50_positive_seeds"] == 5
    assert reading["weighted_hit50_mean_positive"] is True
    assert reading["weighted_hit50_positive_seeds"] == 5
    assert reading["ndcg10_guard_ratio"] == pytest.approx(0.995)
    assert reading["retained_for_final_evaluation"] is True


def test_five_seed_reading_rejects_inconsistent_deep_candidate_gain():
    curve = pd.concat(
        [_curve(seed, favorable=seed != 46) for seed in range(42, 47)],
        ignore_index=True,
    )
    # Make one more seed unfavorable so only 3/5 improve Recall@50.
    mask = (curve.seed == 45) & (curve.model_id == completion.M2_MODEL_ID)
    curve.loc[mask, "recall@50"] = 0.29

    _, _, _, reading = completion.read_five_seed_result(curve)

    assert reading["recall50_positive_seeds"] == 3
    assert reading["retained_for_final_evaluation"] is False


def test_five_seed_reading_rejects_missing_or_duplicate_arm():
    curve = pd.concat([_curve(seed) for seed in range(42, 47)], ignore_index=True)
    with pytest.raises(ValueError, match="seed-model"):
        completion.read_five_seed_result(curve.iloc[:-1])
    with pytest.raises(ValueError, match="seed-model"):
        completion.read_five_seed_result(pd.concat([curve, curve.iloc[[0]]]))


def test_colab_is_pinned_and_runs_the_completion_once():
    notebook = json.loads(
        Path("clv_m2_history_budget_dev5_completion_colab.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "SOURCE_COMMIT" in source
    assert source.count("completion.run(cfg, root=ROOT)") == 1
    assert "seeds == (45, 46)" in source
    assert "final_test" not in source.lower()
