import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_neighbor_conditioned_id_transform_test10 as final10


def test_final_protocol_is_locked_to_two_models_and_ten_paired_seeds(tmp_path):
    cfg = final10.configure_neighbor_conditioned_test10_run(
        out_dir=str(tmp_path)
    )
    summary = final10.preflight_summary(cfg)

    assert cfg.seeds == tuple(range(42, 52))
    assert cfg.epochs == 100
    assert summary["trained_models"] == [
        "m1_64",
        "m2_neighbor_conditioned_id_transform",
    ]
    assert summary["validation_selection"] is False
    assert summary["holdout_evaluation"] is False
    assert summary["automatic_epoch_resume"] is True


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("seeds", (42, 43)),
        ("epochs", 99),
        ("dataset", "hm"),
        ("embedding_dim", 32),
        ("transform_rank", 8),
        ("rho", 0.1),
    ],
)
def test_final_protocol_rejects_post_test_tuning(tmp_path, override, value):
    with pytest.raises(ValueError):
        final10.configure_neighbor_conditioned_test10_run(
            out_dir=str(tmp_path), **{override: value}
        )


def test_base_config_uses_only_merged_train_and_final_test(tmp_path):
    cfg = final10.configure_neighbor_conditioned_test10_run(
        out_dir=str(tmp_path)
    )
    base = final10._base_config(cfg)

    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_USER_INTER"] == 1
    assert base["MIN_ITEM_INTER"] == 1
    assert base["EPOCHS"] == 100


def test_summary_uses_same_seed_paired_differences():
    cfg = final10.NeighborConditionedTest10Config(out_dir="/tmp/test10")
    arms = []
    rows = []
    for seed in cfg.seeds:
        baseline_value = seed / 1000.0
        for model_id, delta in (
            ("m1_64", 0.0),
            ("m2_neighbor_conditioned_id_transform", 0.01),
        ):
            metrics = {"recall@10": baseline_value + delta}
            arm = {
                "seed": seed,
                "model_id": model_id,
                "metrics": metrics,
            }
            arms.append(arm)
            rows.append(
                {"seed": seed, "model_id": model_id, **metrics}
            )
    absolute = pd.DataFrame(rows)

    absolute_summary, paired_seed, paired_summary = final10._summary_tables(
        absolute, arms, cfg
    )

    assert len(absolute_summary) == 2
    assert np.allclose(paired_seed["delta"], 0.01)
    assert paired_summary.loc[0, "mean"] == pytest.approx(0.01)
    assert paired_summary.loc[0, "positive_seed_count"] == 10
