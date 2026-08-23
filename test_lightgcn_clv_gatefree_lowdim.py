import json

import pytest

import lightgcn_clv_gatefree_lowdim as runner


def test_screen_is_one_new_model_on_historical_development_split(tmp_path):
    cfg = runner.configure_gatefree_lowdim_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == ["m2_gatefree_lowdim"]
    assert summary["reused_comparator"] == "m1_64"
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
    }
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["validation_or_epoch_selection"] is False
    assert summary["m2"]["id_dim"] == 64
    assert summary["m2"]["activity_dim"] == 4
    assert summary["m2"]["transaction_value_dim"] == 4
    assert summary["m2"]["explicit_item_features"] is False
    assert summary["m2"]["user_gate"] is False
    assert summary["m2"]["learned_axis_weight"] is False


def test_base_config_preserves_m2_boundaries(tmp_path):
    cfg = runner.configure_gatefree_lowdim_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
    )
    base = runner._base_config(cfg)

    assert base["TIME_CUTOFF"] == 690
    assert base["TRAIN_ON_VAL"] is True
    assert base["TEST_DAYS"] == 7
    assert base["HOLDOUT_DAYS"] == 0
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_USER_INTER"] == 1
    assert base["MIN_ITEM_INTER"] == 1


def _baseline_payload(input_manifest, *, cutoff=690):
    return {
        "code_version": "m2-popularity-controlled-repeatshare-backtest-v1",
        "config": {
            "dataset": "dunnhumby",
            "seed": 42,
            "time_cutoff": cutoff,
            "evaluation_days": 7,
            "epochs": 100,
            "id_dim": 64,
            "n_layers": 2,
            "batch_size": 8192,
            "lr": 5e-4,
            "pref_reg": 1e-3,
        },
        "preflight": {
            "historical_development_split": {
                "train_end_inclusive": 683,
                "evaluation_start_inclusive": 684,
                "evaluation_end_inclusive": 690,
                "original_validation_test_holdout_constructed": False,
            },
            "fixed": {
                "graph": "binary",
                "negative_sampling": "uniform",
                "sample_weighting": False,
                "validation_or_epoch_selection": False,
            },
        },
        "input_manifest": input_manifest,
        "absolute_rows": [
            {
                "model_id": "m1_64",
                "role": "baseline",
                "seed": 42,
                "split": "historical_development_days_684_690",
                "final_epoch": 100,
                "recall@10": 0.1,
                "ndcg@10": 0.2,
            }
        ],
    }


def test_existing_m1_is_reused_only_when_protocol_and_input_match(tmp_path):
    baseline_dir = tmp_path / "old"
    baseline_dir.mkdir()
    manifest = [{"path": "transactions.csv", "sha256": "abc"}]
    path = baseline_dir / "m2_repeatshare_backtest_fixture.json"
    path.write_text(json.dumps(_baseline_payload(manifest)), encoding="utf-8")
    cfg = runner.configure_gatefree_lowdim_run(
        out_dir=str(tmp_path / "new"), baseline_result_dir=str(baseline_dir)
    )

    reused = runner._load_compatible_baseline(cfg, manifest)

    assert reused["model_id"] == "m1_64"
    assert reused["recall@10"] == pytest.approx(0.1)
    assert reused["source_result"] == str(path)


def test_existing_m1_reuse_fails_closed_on_split_mismatch(tmp_path):
    baseline_dir = tmp_path / "old"
    baseline_dir.mkdir()
    manifest = [{"path": "transactions.csv", "sha256": "abc"}]
    (baseline_dir / "m2_repeatshare_backtest_bad.json").write_text(
        json.dumps(_baseline_payload(manifest, cutoff=697)), encoding="utf-8"
    )
    cfg = runner.configure_gatefree_lowdim_run(
        out_dir=str(tmp_path / "new"), baseline_result_dir=str(baseline_dir)
    )

    with pytest.raises(RuntimeError, match="호환되는 M1"):
        runner._load_compatible_baseline(cfg, manifest)
