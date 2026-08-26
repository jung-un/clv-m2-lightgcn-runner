import json

import pandas as pd
import pytest

import lightgcn_clv_history_only_control as runner


def test_preflight_freezes_matched_history_only_control(tmp_path):
    cfg = runner.configure_history_only_control(
        out_dir=str(tmp_path / "control"),
        baseline_result_dir=str(tmp_path / "baseline"),
        full_m2_result_dir=str(tmp_path / "full"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == ["history_only_rho0"]
    assert summary["reused_comparators"] == [
        "m1_64",
        "m2_history_conditioned_lowrank_transform",
    ]
    assert summary["control"]["rho"] == 0.0
    assert summary["control"]["free_user_id_embedding"] is False
    assert summary["control"]["layer0_identity"] == "E_u^(0)=H_u exactly"
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_shared_protocol_keeps_new_item_historical_split(tmp_path):
    cfg = runner.configure_history_only_control(
        out_dir=str(tmp_path / "control"),
        baseline_result_dir=str(tmp_path / "baseline"),
        full_m2_result_dir=str(tmp_path / "full"),
    )
    base = runner._base_config(cfg)

    assert base["TIME_CUTOFF"] == 690
    assert base["TRAIN_ON_VAL"] is True
    assert base["TEST_DAYS"] == 7
    assert base["HOLDOUT_DAYS"] == 0
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_ITEM_INTER"] == 1


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rho": 0.05}, "rho=0.0"),
        ({"seed": 43}, "seed=42"),
        ({"epochs": 99}, "epochs=100"),
        ({"embedding_dim": 96}, "embedding_dim=64"),
    ],
)
def test_unmatched_control_variants_are_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        runner.configure_history_only_control(
            out_dir=str(tmp_path / "control"),
            baseline_result_dir=str(tmp_path / "baseline"),
            full_m2_result_dir=str(tmp_path / "full"),
            **override,
        )


def test_full_result_loader_requires_matching_manifest_and_protocol(tmp_path):
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    payload = {
        "code_version": runner.full_runner.CODE_VERSION,
        "config": {
            "seed": 42,
            "time_cutoff": 690,
            "evaluation_days": 7,
            "epochs": 100,
            "embedding_dim": 64,
            "transform_rank": 4,
            "rho": 0.05,
            "n_layers": 2,
            "batch_size": 8192,
            "lr": 5e-4,
            "pref_reg": 1e-3,
            "input_days": 365,
        },
        "input_manifest": {"data": "same"},
        "absolute_rows": [
            {
                "model_id": runner.FULL_MODEL_ID,
                "recall@10": 0.11,
                "price_purchase_amount_weighted_hit@10": 0.21,
            }
        ],
    }
    result_path = full_dir / "m2_history_conditioned_lowrank_example.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = runner.configure_history_only_control(
        out_dir=str(tmp_path / "control"),
        baseline_result_dir=str(tmp_path / "baseline"),
        full_m2_result_dir=str(full_dir),
    )
    prepared = {
        "manifest": {"data": "same"},
        "baseline": {
            "recall@10": 0.1,
            "price_purchase_amount_weighted_hit@10": 0.2,
        },
    }

    loaded = runner._load_full_m2_result(cfg, prepared)

    assert loaded["metrics"] == {
        "recall@10": 0.11,
        "price_purchase_amount_weighted_hit@10": 0.21,
    }
    assert loaded["source_result"] == str(result_path)


def test_mechanism_reading_separates_base_loss_from_transform_effect():
    metrics = [
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    ]
    rows = [
        {
            "metric": metric,
            "m1_64": 1.0,
            runner.MODEL_ID: 0.95,
            runner.FULL_MODEL_ID: 0.96,
        }
        for metric in metrics
    ]
    rows.append(
        {
            "metric": "price_purchase_amount_weighted_hit@10",
            "m1_64": 1.0,
            runner.MODEL_ID: 0.95,
            runner.FULL_MODEL_ID: 0.97,
        }
    )

    reading = runner._mechanism_reading(pd.DataFrame(rows))

    assert reading["classification"] == (
        "n_v_helps_history_base_but_user_id_loss_remains"
    )
