from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

import clv_dh_m5_ab_adaptive_remaining8 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared


@pytest.fixture
def batch(prepared):
    cfg = s.configure(out_dir=str(prepared["run_dir"]))
    refs, old = {}, {}
    for seed in s.SEEDS:
        r = prepared["references"]["absolute"].assign(seed=seed)
        refs[seed] = {**prepared["references"], "absolute": r,
                      "diagnostics": r[["model_id", "seed"]]}
        if seed in s.REUSED_SEEDS:
            labels = s.base.labels_for(s.seed_config(cfg, seed))
            a = pd.DataFrame([{"model": labels[mid], "model_id": mid, "seed": seed,
                              "source": "new_fit", **dict.fromkeys(refs[seed]["metrics"], 1.01)}
                             for mid in s.base.models_for(s.seed_config(cfg, seed))])
            old[seed] = {"absolute": a, "diagnostics": a[["model_id", "seed"]], "blobs": {}}
    s.base.io._atomic_json(prepared["run_dir"] / "protocol.json", {})
    return {"cfg": cfg, "shared": prepared, "references": refs, "previous": old,
            "run_dir": prepared["run_dir"], "protocol": {}}


def test_only_16_new_fits_and_no_ref_evaluation(batch, monkeypatch):
    calls, evaluations = [], []
    def train(model, p, cfg, mid, seed, store, row_weights=None):
        calls.append((seed, s.base.labels_for(cfg)[mid]))
        assert seed not in (49, 50) and cfg.epochs == 300
        return [{"epoch": e} for e in range(1, 301)]
    def evaluate(model, p):
        evaluations.append(p["cfg"].seeds[0])
        return dict.fromkeys(p["references"]["metrics"], 1.02)
    monkeypatch.setattr(s.base.fixed, "_train", train)
    monkeypatch.setattr(s.base.fixed, "_evaluate", evaluate)
    initial = s.report(batch)
    assert len(initial["absolute"]) == 54 and not initial["complete"]
    assert "summary" not in initial and len(initial["reading"]["missing"]) == 16
    with pytest.raises(ValueError):
        s.context(batch, 49)
    with pytest.raises(RuntimeError, match="미완료"):
        s.export_zip(batch, batch["run_dir"] / "premature.zip")
    # Simulate restart after one arm was saved, then run and rerun whole batch.
    ctx = s.context(batch, 42)
    s.base.run_arm(ctx, s.base.models_for(ctx["cfg"])[0])
    assert len(s.report(batch)["reading"]["missing"]) == 15
    result = s.run(batch)
    s.run(batch)
    assert calls == [(seed, model) for seed in s.NEW_SEEDS for model in ("M5_A", "M5_B")]
    assert len(evaluations) == 16
    assert result["complete"] and len(result["absolute"]) == 70
    assert set(result["summary"].n_seeds) == {10}
    assert len(result["summary"]) == 7 * 142
    assert len(result["comparison"]) == 12 * 142
    assert len(result["paired_comparison"]) == 12 * 142 * 10
    output = s.export_zip(batch, batch["run_dir"] / "complete.zip")
    with ZipFile(output) as z:
        assert "core_mean_std.csv" in z.namelist()
        assert len([n for n in z.namelist() if n.startswith("arms/")]) == 16


def test_mean_std_and_paired_difference_are_independent_of_row_order():
    rows = [{"model": m, "seed": seed, "metric": i + 1 + (2 if m == "M5_A" else 0)}
            for m in s.ORDER for i, seed in enumerate(s.SEEDS)]
    a = pd.DataFrame(rows).sample(frac=1, random_state=13)
    summary, comparison, paired = s.aggregate(a, ["metric"])
    row = summary[summary.model == "M5_A"].iloc[0]
    assert row["mean"] == 7.5
    assert row["std"] == pytest.approx(np.sqrt(82.5/9))
    r = comparison[(comparison.model == "M5_A") & (comparison.reference == "M1")].iloc[0]
    assert r.paired_delta_mean == 2 and r.paired_delta_std == 0
    assert r.ratio_of_means_pct == pytest.approx((7.5/5.5-1)*100)
    assert r.paired_pct_mean != pytest.approx(r.ratio_of_means_pct)
    assert r.positive_delta_seeds == 10
    assert set(paired.seed) == set(s.SEEDS)
    assert s.aggregate(a.iloc[:-1], ["metric"])[0].empty
    with pytest.raises(RuntimeError, match="중복"):
        s.aggregate(pd.concat([a, a.iloc[:1]]), ["metric"])


def test_unverified_prior_stops_before_data_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(s.base, "load_references", lambda cfg: {})
    monkeypatch.setattr(s.base, "prepare", lambda *a: pytest.fail("expensive prep started"))
    with pytest.raises(RuntimeError, match="원본"):
        s.prepare(s.configure(seed49_reference=str(tmp_path)))


def test_reference_and_output_must_differ(tmp_path):
    with pytest.raises(ValueError):
        s.load_sources(s.configure(out_dir=str(tmp_path), m2_reference=str(tmp_path)))
