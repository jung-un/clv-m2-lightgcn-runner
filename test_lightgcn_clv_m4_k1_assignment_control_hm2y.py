import importlib

import numpy as np
import pytest


MODULE = "lightgcn_clv_m4_k1_assignment_control_hm2y"


def _hm():
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError:
        pytest.fail(f"H&M M4 assignment runner가 아직 없습니다: {MODULE}")


def _cfg(tmp_path):
    hm = _hm()
    return hm.configure_hm2y_m4_assignment_screen(
        out_dir=str(tmp_path / "results")
    )


def test_contract_is_hm_validation_only_three_arm_k1(tmp_path):
    hm = _hm()
    cfg = _cfg(tmp_path)
    summary = hm.preflight_summary(cfg)

    assert (cfg.dataset, cfg.seed, cfg.negative_count) == ("hm", 42, 1)
    assert cfg.batch_size == 131_072
    assert cfg.positive_weight_lambda == 0.5
    assert summary["split"] == hm.SPLIT_LABEL
    assert summary["trained_models"] == list(hm.MODEL_IDS)
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["checkpointing"]["save_after_each_completed_epoch"] is True
    assert summary["checkpointing"]["atomic_replace"] is True


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("negative_count", 5),
        ("epochs", 99),
        ("batch_size", 65_536),
        ("positive_weight_lambda", 0.25),
        ("include_shuffle", False),
    ],
)
def test_contract_rejects_protocol_changes(tmp_path, override, value):
    hm = _hm()
    with pytest.raises(ValueError, match="H&M M4 K=1"):
        hm.configure_hm2y_m4_assignment_screen(
            out_dir=str(tmp_path / "results"), **{override: value}
        )


def test_arm_specs_change_only_q_c_for_the_two_m4_arms():
    hm = _hm()
    actual = np.array([0.2, 0.8], dtype=np.float32)
    shuffled = np.array([0.8, 0.2], dtype=np.float32)
    prepared = {
        "q_c": actual,
        "q_c_shuffle": {"q_c": shuffled},
        "m2_actual": {
            "q_n": np.array([0.1, 0.9], dtype=np.float32),
            "q_v": np.array([0.3, 0.7], dtype=np.float32),
            "q_c": actual,
            "clv_valid": np.array([True, True]),
        },
    }

    specs = hm.arm_specifications(prepared)

    assert [spec["model_id"] for spec in specs] == list(hm.MODEL_IDS)
    assert [(spec["rho"], spec["improvement"]) for spec in specs] == [
        (0.0, None),
        (0.0, "original"),
        (0.0, "original"),
    ]
    np.testing.assert_array_equal(specs[1]["q_c"], actual)
    np.testing.assert_array_equal(specs[2]["q_c"], shuffled)
    assert all(spec["split"] == hm.SPLIT_LABEL for spec in specs)


def test_arm_prepared_replaces_only_the_m4_q_c_assignment():
    hm = _hm()
    actual = np.array([0.2, 0.8], dtype=np.float32)
    shuffled = np.array([0.8, 0.2], dtype=np.float32)
    q_n = np.array([0.1, 0.9], dtype=np.float32)
    prepared = {
        "q_c": actual,
        "m2_actual": {
            "q_n": q_n,
            "q_v": np.array([0.3, 0.7], dtype=np.float32),
            "q_c": actual,
            "clv_valid": np.array([True, True]),
        },
    }
    spec = {"q_c": shuffled}

    arm = hm.arm_prepared(prepared, spec)

    np.testing.assert_array_equal(arm["q_c"], shuffled)
    np.testing.assert_array_equal(arm["m2_actual"]["q_c"], shuffled)
    np.testing.assert_array_equal(prepared["q_c"], actual)
    np.testing.assert_array_equal(prepared["m2_actual"]["q_n"], q_n)


def test_attribution_requires_actual_to_beat_m1_and_shuffle():
    hm = _hm()

    def metrics(economic, accuracy=1.0):
        return {
            "recall@10": 0.0114 * accuracy,
            "ndcg@10": 0.0077 * accuracy,
            "recall@20": 0.018 * accuracy,
            "ndcg@20": 0.010 * accuracy,
            "recall@50": 0.034 * accuracy,
            "ndcg@50": 0.014 * accuracy,
            "price_purchase_amount_weighted_hit@10": economic,
            "vndcg@10": economic * 8,
        }

    rows = {
        hm.M1_MODEL_ID: metrics(0.00088),
        hm.M4_ACTUAL_MODEL_ID: metrics(0.00092),
        hm.M4_SHUFFLED_MODEL_ID: metrics(0.00090),
    }
    cvs = {hm.M4_ACTUAL_MODEL_ID: 0.11, hm.M4_SHUFFLED_MODEL_ID: 0.10}

    passed = hm.attribution_reading(
        rows,
        actual_vs_shuffle_top10_change_share=0.05,
        row_weight_cvs=cvs,
    )
    rows[hm.M4_SHUFFLED_MODEL_ID] = metrics(0.00093)
    failed = hm.attribution_reading(
        rows,
        actual_vs_shuffle_top10_change_share=0.05,
        row_weight_cvs=cvs,
    )

    assert passed["attribution_pass"] is True
    assert failed["attribution_pass"] is False


def test_training_metadata_uses_the_hm_split_label():
    hm = _hm()
    training = importlib.import_module(
        "lightgcn_clv_m5_k1_m4_improvement_screen"
    )

    stage, split = training.run_labels({"split": hm.SPLIT_LABEL})

    assert stage == "hm2y_development_train"
    assert split == hm.SPLIT_LABEL
