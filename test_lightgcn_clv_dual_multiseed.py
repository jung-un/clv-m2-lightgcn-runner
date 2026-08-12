import dataclasses
import json

import pandas as pd
import pytest


def _rows(revenue_deltas=(0.1, 0.1, 0.1), accuracy_ratio=1.0):
    rows = []
    for seed, revenue_delta in zip((42, 43, 44), revenue_deltas):
        baseline = {
            f"{metric}@{k}": 1.0
            for metric in ("recall", "ndcg")
            for k in (10, 20, 50)
        }
        rows.append(
            {
                "seed": seed,
                "model_id": "m1",
                "revenue@10": 1.0,
                **baseline,
            }
        )
        rows.append(
            {
                "seed": seed,
                "model_id": "dual_clv_fixed",
                "revenue@10": 1.0 + revenue_delta,
                **{key: value * accuracy_ratio for key, value in baseline.items()},
            }
        )
    return pd.DataFrame(rows)


def test_multiseed_presets_are_frozen_and_exclude_protected_runs(tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    dun = multiseed.configure_multiseed_validation(
        "dunnhumby", tmp_path / "dun.json"
    )
    hm = multiseed.configure_multiseed_validation(
        "hm", tmp_path / "hm.json", short_hm=True
    )
    assert (dun.gate_shape, dun.fixed_lambda) == ("equal", 2.0)
    assert (hm.window_days, hm.gate_shape, hm.fixed_lambda) == (60, "high", 1.0)
    assert dun.new_seeds == hm.new_seeds == (43, 44)
    assert dun.model_ids == hm.model_ids == ("m1", "dual_clv_fixed")
    assert dun.eval_test is hm.eval_test is False
    assert dun.eval_holdout is hm.eval_holdout is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"new_seeds": (42, 43)}, "43, 44"),
        ({"eval_test": True}, "validation-only"),
        ({"eval_holdout": True}, "validation-only"),
        ({"model_ids": ("m1", "dual_clv_fixed", "dual_adapter_only")}, "두 모형"),
        ({"gate_shape": "high"}, "Dunnhumby"),
        ({"fixed_lambda": 1.0}, "Dunnhumby"),
    ],
)
def test_multiseed_config_rejects_protocol_changes(changes, message, tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    cfg = multiseed.configure_multiseed_validation(
        "dunnhumby", tmp_path / "dun.json"
    )
    with pytest.raises(ValueError, match=message):
        multiseed.validate_multiseed_config(dataclasses.replace(cfg, **changes))


def test_hm_full_period_is_rejected_for_this_runner(tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    with pytest.raises(ValueError, match="60일"):
        multiseed.configure_multiseed_validation(
            "hm", tmp_path / "hm.json", short_hm=False
        )


def test_reproducibility_decision_passes_all_three_frozen_conditions():
    from lightgcn_clv_dual_multiseed import reproducibility_decision

    decision = reproducibility_decision(_rows())
    assert decision["success"] is True
    assert decision["positive_revenue_seed_count"] == 3
    assert decision["mean_revenue_delta"] == pytest.approx(0.1)
    assert decision["failed_conditions"] == []


@pytest.mark.parametrize(
    ("rows", "failed"),
    [
        (_rows((-0.2, 0.05, 0.05)), "mean_revenue_delta_positive"),
        (_rows((0.2, -0.01, -0.01)), "positive_revenue_in_at_least_two_seeds"),
        (_rows(accuracy_ratio=0.98), "six_accuracy_mean_ratios_at_least_0.99"),
    ],
)
def test_reproducibility_decision_reports_each_failure(rows, failed):
    from lightgcn_clv_dual_multiseed import reproducibility_decision

    decision = reproducibility_decision(rows)
    assert decision["success"] is False
    assert failed in decision["failed_conditions"]


def _fake_seed_result(seed, revenue_delta=0.1):
    accuracy = {
        f"{metric}@{k}": 1.0
        for metric in ("recall", "ndcg")
        for k in (10, 20, 50)
    }
    baseline = {
        "seed": seed,
        "model_id": "m1",
        "split": "val",
        "gate_shape": "none",
        "lambda": 0.0,
        "revenue@10": 1.0,
        **accuracy,
    }
    model = {
        **baseline,
        "model_id": "dual_clv_fixed",
        "gate_shape": "equal",
        "lambda": 2.0,
        "revenue@10": 1.0 + revenue_delta,
    }
    per_user = {
        name: [1.0, 2.0, 3.0]
        for name in ("recall", "ndcg", "revenue", "arp")
    }
    model_per_user = {name: [value + revenue_delta for value in values] for name, values in per_user.items()}
    return {
        "rows": [baseline, model],
        "baseline_per_user": per_user,
        "model_per_user": model_per_user,
        "eval_users": [1, 2, 3],
        "checkpoints": {},
        "checkpoint_sha256": {},
        "training": {},
    }


def test_run_trains_only_new_seeds_and_never_starts_followup(monkeypatch, tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    source = tmp_path / "seed42.json"
    source.write_text(json.dumps({"fixture": True}))
    cfg = multiseed.configure_multiseed_validation(
        "dunnhumby", source, out_dir=tmp_path / "out"
    )
    calls = []
    monkeypatch.setattr(
        multiseed,
        "_load_and_validate_seed42_payload",
        lambda _cfg: {"result_fingerprint": "fixture"},
    )
    monkeypatch.setattr(
        multiseed,
        "_load_seed42_evaluation",
        lambda _cfg, _payload: _fake_seed_result(42),
    )

    def fake_new_seed(_cfg, _payload, seed):
        calls.append(seed)
        return _fake_seed_result(seed)

    monkeypatch.setattr(multiseed, "_run_new_seed", fake_new_seed)
    monkeypatch.setattr(
        multiseed,
        "_persist",
        lambda _cfg, _payload, results, decision: pd.DataFrame(
            [row for result in results for row in result["rows"]]
        ),
    )

    frame = multiseed.run_multiseed_validation(cfg)
    assert calls == [43, 44]
    assert set(frame.seed) == {42, 43, 44}
    assert set(frame.model_id) == {"m1", "dual_clv_fixed"}


def test_persist_writes_six_rows_delta_decision_and_provenance(tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    source = tmp_path / "seed42.json"
    source.write_text("{}")
    cfg = multiseed.configure_multiseed_validation(
        "dunnhumby", source, out_dir=tmp_path / "out"
    )
    results = [_fake_seed_result(seed) for seed in (42, 43, 44)]
    payload = {
        "result_fingerprint": "seed42",
        "source_revision": "old",
        "input_manifest": {"transactions": {"bytes": 1, "sha256": "a"}},
    }
    decision = multiseed.reproducibility_decision(
        pd.DataFrame([row for result in results for row in result["rows"]])
    )

    frame = multiseed._persist(cfg, payload, results, decision)
    paths = frame.attrs["result_paths"]
    assert len(pd.read_csv(paths["csv"])) == 6
    delta = pd.read_csv(paths["delta_csv"])
    assert set(delta["scope"]) == {"seed", "three_seed"}
    decision_table = pd.read_csv(paths["decision_csv"])
    assert set(decision_table["condition"]) == set(decision["conditions"])
    saved = json.loads(open(paths["json"], encoding="utf-8").read())
    assert saved["original_seed42_result_fingerprint"] == "seed42"
    assert saved["reproducibility_decision"]["success"] is True
    assert saved["interpretation"]["hm_two_year_executed"] is False


def test_new_seed_reuses_exact_seed42_training_budget(monkeypatch, tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    cfg = multiseed.configure_multiseed_validation(
        "dunnhumby", tmp_path / "seed42.json", out_dir=tmp_path / "out"
    )
    source_config = {
        field.name: field.default
        for field in dataclasses.fields(multiseed.moe.MoEConfig)
    }
    source_config.update(
        dataset="dunnhumby",
        seed_list=[42],
        max_epochs=77,
        encoder_epochs=66,
        lambda_eval=[0.0, 2.0],
        out_dir="old",
    )
    captured = {}

    def stop_after_config(run_cfg, seed):
        captured["cfg"] = run_cfg
        captured["seed"] = seed
        raise RuntimeError("stop")

    monkeypatch.setattr(multiseed.dual, "_prepare", stop_after_config)
    with pytest.raises(RuntimeError, match="stop"):
        multiseed._run_new_seed(cfg, {"config": source_config}, 43)

    assert captured["seed"] == 43
    assert captured["cfg"].max_epochs == 77
    assert captured["cfg"].encoder_epochs == 66
    assert captured["cfg"].seed_list == (43,)
    assert captured["cfg"].lambda_eval == (2.0,)
    assert captured["cfg"].window_days is None
