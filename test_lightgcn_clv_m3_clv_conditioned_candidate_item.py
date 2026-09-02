import pandas as pd
import pytest

import lightgcn_clv_m3_clv_conditioned_candidate_item as runner


def test_preflight_locks_direct_candidate_item_screen(tmp_path):
    cfg = runner.configure_clv_candidate_item_run(
        out_dir=str(tmp_path / "m3"),
        baseline_result_dir=str(tmp_path / "m1"),
    )
    summary = runner.preflight_summary(cfg)
    assert summary["trained_models"] == [
        runner.GENERAL_ID,
        runner.ACTUAL_ID,
        runner.SHUFFLE_ID,
    ]
    assert summary["reused_comparator"] == runner.M1_ID
    assert summary["m3"]["historical_clv_proxy"] == "N_hat * V_hat"
    assert summary["m3"]["candidate_train_pairs_excluded"] is True
    assert summary["m3"]["item_minimum_distinct_user_support"] == 5
    assert summary["m3"]["max_candidate_items_per_user"] == 100
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["reading_rule"]["accuracy_guardrails"] is False


def test_preflight_locks_common_support_weight_screen(tmp_path):
    cfg = runner.configure_clv_candidate_item_common_support_run(
        out_dir=str(tmp_path / "m3-v2"),
        baseline_result_dir=str(tmp_path / "m1"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["code_version"] == runner.COMMON_SUPPORT_CODE_VERSION
    assert summary["trained_models"] == [
        runner.M1_ID,
        runner.COMMON_SUPPORT_GENERAL_ID,
        runner.COMMON_SUPPORT_ACTUAL_ID,
        runner.COMMON_SUPPORT_SHUFFLE_ID,
    ]
    assert summary["reused_comparator"] is None
    assert summary["trained_comparator"] == runner.M1_ID
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 690,
        "evaluation_start_inclusive": 691,
        "evaluation_end_inclusive": 697,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["m3"]["candidate_support"] == (
        "pooled top candidates fixed identically across all arms"
    )
    assert summary["m3"]["positive_excess_clipping"] is False


def test_common_support_config_rejects_old_relation_mode(tmp_path):
    with pytest.raises(ValueError, match="common-support"):
        runner.configure_clv_candidate_item_common_support_run(
            out_dir=str(tmp_path / "m3-v2"),
            baseline_result_dir=str(tmp_path / "m1"),
            relation_mode=runner.RELATION_MODE_POSITIVE_EXCESS,
        )


def test_common_support_config_rejects_seen_development_interval(tmp_path):
    with pytest.raises(ValueError, match="time_cutoff=697"):
        runner.configure_clv_candidate_item_common_support_run(
            out_dir=str(tmp_path / "m3-v2"),
            time_cutoff=690,
        )


def test_supplemental_preflight_locks_graph_and_blocks_unapproved_evaluation(
    tmp_path,
):
    cfg = runner.configure_clv_candidate_item_supplemental_run(
        out_dir=str(tmp_path / "m3-supplemental"),
        evaluation_authorized=False,
    )
    summary = runner.preflight_summary(cfg)

    assert summary["code_version"] == runner.SUPPLEMENTAL_CODE_VERSION
    assert summary["trained_models"] == [
        runner.M1_ID,
        runner.SUPPLEMENTAL_GENERAL_ID,
        runner.SUPPLEMENTAL_ACTUAL_ID,
        runner.SUPPLEMENTAL_SHUFFLE_ID,
    ]
    assert summary["m3"]["base_candidate_items"] == 100
    assert summary["m3"]["supplemental_candidate_items"] == 20
    assert summary["m3"]["base_mass"] == pytest.approx(5 / 6)
    assert summary["m3"]["supplemental_mass"] == pytest.approx(1 / 6)
    assert summary["performance_evaluation_authorized"] is False
    with pytest.raises(RuntimeError, match="evaluation protocol approval"):
        runner.run_clv_candidate_item_supplemental(cfg)


def test_supplemental_config_rejects_seen_evaluation_boundary(tmp_path):
    with pytest.raises(ValueError, match="already-seen DAY 684--704"):
        runner.configure_clv_candidate_item_supplemental_run(
            out_dir=str(tmp_path / "m3-supplemental"),
            time_cutoff=697,
            evaluation_authorized=True,
            evaluation_protocol_label="approved-new-test",
        )


def test_supplemental_authorized_config_requires_protocol_label(tmp_path):
    with pytest.raises(ValueError, match="evaluation_protocol_label"):
        runner.configure_clv_candidate_item_supplemental_run(
            out_dir=str(tmp_path / "m3-supplemental"),
            time_cutoff=711,
            evaluation_authorized=True,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"time_cutoff": 697},
        {"gamma": 0.1},
        {"category_kappa": 10.0},
        {"item_kappa": 10.0},
        {"item_min_support_users": 3},
        {"cross_fit_folds": 4},
        {"max_candidate_items": 50},
        {"epochs": 50},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(tmp_path, override):
    with pytest.raises(ValueError, match="candidate-item screen"):
        runner.configure_clv_candidate_item_run(
            out_dir=str(tmp_path / "m3"),
            baseline_result_dir=str(tmp_path / "m1"),
            **override,
        )


def _metrics(multiplier):
    return {
        "recall@10": 0.01 * multiplier,
        "ndcg@10": 0.011 * multiplier,
        "recall@20": 0.02 * multiplier,
        "ndcg@20": 0.021 * multiplier,
        "recall@50": 0.04 * multiplier,
        "ndcg@50": 0.03 * multiplier,
        "price_purchase_amount_weighted_hit@10": 0.3 * multiplier,
        "mean_recommended_price_percentile@10": 0.25,
    }


def test_clv_attribution_requires_actual_to_beat_all_references():
    rows = [
        {"model_id": runner.M1_ID, **_metrics(1.00)},
        {"model_id": runner.GENERAL_ID, **_metrics(1.01)},
        {"model_id": runner.ACTUAL_ID, **_metrics(1.02)},
        {"model_id": runner.SHUFFLE_ID, **_metrics(1.005)},
    ]
    reading = runner.attribution_reading(pd.DataFrame(rows))
    assert reading["clv_attribution_supported"] is True
    assert reading["six_metric_balance_actual_vs_m1"] > 1
    assert reading["six_metric_balance_actual_vs_general_candidate_relation"] > 1
    assert reading["six_metric_balance_actual_vs_shuffle"] > 1

    rows[-1] = {"model_id": runner.SHUFFLE_ID, **_metrics(1.03)}
    assert runner.attribution_reading(pd.DataFrame(rows))[
        "clv_attribution_supported"
    ] is False


def test_common_support_attribution_uses_common_support_model_ids():
    model_ids = {
        "m1": runner.M1_ID,
        "general": runner.COMMON_SUPPORT_GENERAL_ID,
        "actual": runner.COMMON_SUPPORT_ACTUAL_ID,
        "shuffle": runner.COMMON_SUPPORT_SHUFFLE_ID,
    }
    rows = [
        {"model_id": model_ids["m1"], **_metrics(1.00)},
        {"model_id": model_ids["general"], **_metrics(1.01)},
        {"model_id": model_ids["actual"], **_metrics(1.02)},
        {"model_id": model_ids["shuffle"], **_metrics(1.005)},
    ]

    reading = runner.attribution_reading(pd.DataFrame(rows), model_ids)

    assert reading["clv_attribution_supported"] is True


def test_supplemental_attribution_requires_candidate_truth_advantage():
    model_ids = {
        "m1": runner.M1_ID,
        "general": runner.SUPPLEMENTAL_GENERAL_ID,
        "actual": runner.SUPPLEMENTAL_ACTUAL_ID,
        "shuffle": runner.SUPPLEMENTAL_SHUFFLE_ID,
    }
    rows = [
        {"model_id": model_ids["m1"], **_metrics(1.00)},
        {"model_id": model_ids["general"], **_metrics(1.01)},
        {"model_id": model_ids["actual"], **_metrics(1.03)},
        {"model_id": model_ids["shuffle"], **_metrics(1.02)},
    ]

    failed = runner.attribution_reading(
        pd.DataFrame(rows),
        model_ids,
        candidate_truth_hits={
            runner.ARM_ACTUAL: 10,
            runner.ARM_SHUFFLE: 10,
            runner.ARM_GENERAL: 9,
        },
    )
    assert failed["performance_balance_passed"] is True
    assert failed["candidate_truth_advantage_passed"] is False
    assert failed["clv_attribution_supported"] is False

    passed = runner.attribution_reading(
        pd.DataFrame(rows),
        model_ids,
        candidate_truth_hits={
            runner.ARM_ACTUAL: 11,
            runner.ARM_SHUFFLE: 10,
            runner.ARM_GENERAL: 9,
        },
    )
    assert passed["clv_attribution_supported"] is True
