import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m3_mass_preserving as M3


def test_default_is_single_seed_test_only_without_holdout_evaluation(tmp_path):
    cfg = M3.configure_m3_clv_influence_test_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg.seeds == (42,)
    summary = M3.preflight_summary(cfg)
    assert summary["training_data"] == "former train + validation"
    assert summary["validation_constructed"] is False
    assert summary["validation_selection"] is False
    assert summary["early_stopping"] is False
    assert summary["holdout_evaluation"] is False
    assert "ignored" in summary["post_test_rows"]
    assert summary["planned_full_seeds"] == list(range(42, 52))
    assert summary["models"] == list(M3.MODEL_ORDER)


def test_base_config_merges_validation_and_preserves_fixed_test_boundary(tmp_path):
    cfg = M3.configure_m3_clv_influence_test_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    base = M3._base_config(cfg)
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["HOLDOUT_DAYS"] == 7
    assert base["EPOCHS"] == 100
    assert base["EARLY_STOP"] == 100
    assert base["MIN_USER_INTER"] == base["MIN_ITEM_INTER"] == 1
    assert base["GRAPH_MODE"] == "binary"
    assert base["LOSS_MODE"] == "plain"
    assert base["NEG_MODE"] == "uniform"


def test_full_professor_requested_seed_set_is_supported(tmp_path):
    cfg = M3.configure_m3_clv_influence_test_run(
        out_dir=str(tmp_path / "dunnhumby"), seeds=M3.FULL_SEEDS
    )
    assert cfg.seeds == tuple(range(42, 52))
    assert M3.preflight_summary(cfg)["current_scope"] == "multi-seed run"


@pytest.mark.parametrize("seeds", [(), (41,), (42, 42), (43, 42)])
def test_seed_scope_fails_closed(tmp_path, seeds):
    with pytest.raises(ValueError, match="seeds"):
        M3.configure_m3_clv_influence_test_run(
            out_dir=str(tmp_path / "dunnhumby"), seeds=seeds
        )


def _fake_arms(seeds=(42,)):
    arms = []
    for seed in seeds:
        for index, model_id in enumerate(M3.MODEL_ORDER):
            arms.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "role": "baseline" if index == 0 else "control",
                    "final_epoch": 100,
                    "metrics": {
                        "recall@10": 1.0 + index * 0.01 + (seed - 42) * 0.001,
                        "price_purchase_amount_weighted_hit@10": 2.0 + index * 0.02,
                    },
                }
            )
    return arms


def test_single_seed_summary_does_not_invent_variance_or_interval():
    arms = _fake_arms()
    absolute = M3._absolute_rows(arms)
    summary, paired, paired_summary = M3._summary_tables(absolute, arms, (42,))
    assert set(summary["n_seeds"]) == {1}
    assert summary["sd"].isna().all()
    assert summary["lo"].isna().all()
    assert summary["hi"].isna().all()
    assert set(paired["seed"]) == {42}
    assert paired_summary["sd"].isna().all()


def test_ten_seed_summary_reports_means_and_paired_variation():
    seeds = M3.FULL_SEEDS
    arms = _fake_arms(seeds)
    absolute = M3._absolute_rows(arms)
    summary, paired, paired_summary = M3._summary_tables(absolute, arms, seeds)
    assert set(summary["n_seeds"]) == {10}
    assert np.isfinite(summary["sd"]).all()
    assert len(paired) > 0
    assert set(paired_summary["n_seeds"]) == {10}
    assert np.isfinite(paired_summary["lo"]).all()
    assert np.isfinite(paired_summary["hi"]).all()


def test_runner_has_no_test_based_screening_decision():
    assert not hasattr(M3, "screening_decision")
    assert not hasattr(M3, "run_experiment")
