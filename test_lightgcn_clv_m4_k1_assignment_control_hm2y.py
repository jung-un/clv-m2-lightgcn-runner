import importlib
import json
from pathlib import Path

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


@pytest.mark.parametrize("seed", [43, 44])
def test_frozen_replication_allows_only_prespecified_followup_seeds(tmp_path, seed):
    hm = _hm()
    cfg = hm.configure_hm2y_m4_assignment_screen(
        out_dir=str(tmp_path / "results"), seed=seed, shuffle_seed=seed
    )
    summary = hm.preflight_summary(cfg)

    assert cfg.seed == seed
    assert cfg.shuffle_seed == seed
    assert summary["staged_replication"]["seeds"] == [42, 43, 44]
    assert summary["staged_replication"]["pass_rule"] == (
        "at least 2 of 3 seeds pass the frozen single-seed attribution rule"
    )


def test_replication_rejects_unregistered_or_mismatched_shuffle_seed(tmp_path):
    hm = _hm()
    with pytest.raises(ValueError, match="seed"):
        hm.configure_hm2y_m4_assignment_screen(
            out_dir=str(tmp_path / "results"), seed=45, shuffle_seed=45
        )
    with pytest.raises(ValueError, match="shuffle_seed"):
        hm.configure_hm2y_m4_assignment_screen(
            out_dir=str(tmp_path / "results"), seed=43, shuffle_seed=42
        )


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


def test_single_arm_runner_rejects_unknown_model_before_preparation(tmp_path):
    hm = _hm()
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match="selected_model_id"):
        hm.run_hm2y_m4_assignment_arm(cfg, "unknown_arm")


def test_colab_pins_reviewed_source_and_explains_epoch_resume():
    path = Path(
        "clv_m4_personalized_positive_weight_k1_assignment_control_"
        "hm2y_colab.ipynb"
    )
    if not path.exists():
        pytest.fail(f"H&M M4 Colab이 아직 없습니다: {path}")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "REVIEWED_SHA = '4cb1ae12849825975f9d89e035353e720fa130cc'" in source
    assert source.count("run_hm2y_m4_assignment_screen(cfg)") == 1
    assert "summary['trained_models'] == list(hm_screen.MODEL_IDS)" in source
    assert "summary['fixed']['final_test_constructed'] is False" in source
    assert "summary['checkpointing']['save_after_each_completed_epoch'] is True" in source
    assert "마지막으로 완료된 epoch 다음" in source
    assert "rm -rf" not in source


@pytest.mark.parametrize("seed", [43, 44])
def test_replication_colabs_pin_seed_and_reviewed_source(seed):
    path = Path(
        "clv_m4_personalized_positive_weight_k1_assignment_control_"
        f"hm2y_seed{seed}_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "REVIEWED_SHA = '3188b01c360d54e78338eaebdf2a8ecb6a6d52d8'" in source
    assert f"seed={seed}, shuffle_seed={seed}" in source
    assert "summary['staged_replication']['seeds'] == [42, 43, 44]" in source
    assert source.count("run_hm2y_m4_assignment_screen(cfg)") == 1
    assert "마지막으로 완료된 epoch 다음" in source
    assert "rm -rf" not in source


@pytest.mark.parametrize(
    ("label", "model_id"),
    [
        ("m1", "m1_bpr_k1_hm2y_m4_assignment_control"),
        ("actual", "m4_personalized_positive_weight_actual_qc_bpr_k1_hm2y"),
        (
            "shuffle",
            "m4_personalized_positive_weight_degree_matched_qc_shuffle_bpr_k1_hm2y",
        ),
    ],
)
def test_seed44_parallel_colabs_keep_original_checkpoint_identity(label, model_id):
    path = Path(
        "clv_m4_personalized_positive_weight_k1_assignment_control_"
        f"hm2y_seed44_{label}_parallel_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "REVIEWED_SHA = '3188b01c360d54e78338eaebdf2a8ecb6a6d52d8'" in source
    assert f"SELECTED_MODEL_ID = '{model_id}'" in source
    assert "hm_screen._prepare(cfg)" in source
    assert "hm_screen.training._run_arm" in source
    assert "run_hm2y_m4_assignment_screen(cfg)" not in source
    assert "rm -rf" not in source


def test_seed44_parallel_aggregate_only_uses_completed_arm_cache():
    path = Path(
        "clv_m4_personalized_positive_weight_k1_assignment_control_"
        "hm2y_seed44_parallel_aggregate_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "REVIEWED_SHA = '3188b01c360d54e78338eaebdf2a8ecb6a6d52d8'" in source
    assert source.count("run_hm2y_m4_assignment_screen(cfg)") == 1
    assert "세 arm이 모두 완료된 뒤에만" in source
