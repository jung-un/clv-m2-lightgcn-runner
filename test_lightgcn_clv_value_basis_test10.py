import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import lightgcn_clv_value_basis_test10 as test10run


def _cfg(**overrides):
    return test10run.configure_value_basis_test10(
        out_dir="/tmp/value_basis_test10", **overrides
    )


def test_protocol_is_ten_seeds_single_negative_and_no_selection():
    summary = test10run.preflight_summary(_cfg())

    assert summary["seeds"] == list(range(42, 52))
    assert summary["validation_selection"] is False
    assert summary["test_evaluations_per_arm"] == 1
    assert summary["loss"]["negative_count"] == 1
    assert summary["loss"]["hard_negative"] is False
    assert summary["loss"]["row_weighting"] is False
    assert summary["m4_present"] is False
    assert summary["trained_models"] == list(test10run.MODEL_IDS)


def test_config_fails_closed_on_any_protocol_change():
    for field, bad in (
        ("epochs", 50),
        ("negative_count", 5),
        ("rho", 0.05),
        ("seeds", (42,)),
        ("n_layers", 3),
    ):
        with pytest.raises(ValueError):
            _cfg(**{field: bad})


def test_three_arms_cross_the_value_basis_with_the_assignment():
    specs = test10run.arm_specifications()

    assert [
        (spec["model_id"], spec["value_basis"], spec["assignment"]) for spec in specs
    ] == [
        (test10run.M1_MODEL_ID, False, "inactive"),
        (test10run.M2_MODEL_ID, True, "observed_clv"),
        (test10run.M2_SHUFFLE_MODEL_ID, True, "degree_matched_clv_shuffle"),
    ]


def test_run_hash_survives_a_new_arm_but_not_a_model_change(tmp_path, monkeypatch):
    cfg = _cfg()
    before = test10run._config_hash(cfg, "input-hash")

    # Appending an arm later must not invalidate the completed M1/M2 seeds.
    monkeypatch.setattr(
        test10run, "MODEL_IDS", test10run.MODEL_IDS + ("m4_future",), raising=True
    )
    assert test10run._config_hash(cfg, "input-hash") == before

    # A different model definition must invalidate them.
    monkeypatch.setattr(test10run, "file_sha256", lambda path: "changed")
    assert test10run._config_hash(cfg, "input-hash") != before


def test_run_hash_changes_with_the_input_manifest():
    cfg = _cfg()
    assert test10run._config_hash(cfg, "a") != test10run._config_hash(cfg, "b")


def _prepared_stub(moved: bool = True, same_bin: bool = True) -> dict:
    q_n = np.array([0.1, 0.9], dtype=np.float32)
    q_v = np.array([0.2, 0.8], dtype=np.float32)
    q_c = np.array([0.3, 0.7], dtype=np.float32)
    valid = np.array([True, True])
    source = np.array([1, 0]) if moved else np.array([0, 1])
    degree_bin = np.array([0, 0]) if same_bin else np.array([0, 1])
    return {
        "clv_valid": valid,
        "degree_bin": degree_bin,
        "m2_actual": {"q_n": q_n, "q_v": q_v, "q_c": q_c, "clv_valid": valid},
        "m2_shuffle": {
            "q_n": q_n[source],
            "q_v": q_v[source],
            "q_c": q_c[source],
            "clv_valid": valid[source],
            "source_user": source,
        },
    }


def test_shuffle_diagnostics_require_a_within_stratum_permutation():
    diagnostics = test10run._shuffle_diagnostics(_prepared_stub())

    assert diagnostics["moved_user_share"] == 1.0
    assert diagnostics["same_degree_bin"] is True
    assert diagnostics["q_c_multiset_preserved"] is True

    with pytest.raises(RuntimeError, match="순열 불변식"):
        test10run._shuffle_diagnostics(_prepared_stub(same_bin=False))


def _arm(model_id: str, seed: int, economic: float, accuracy: float) -> dict:
    return {
        "model_id": model_id,
        "role": "model",
        "assignment": "observed_clv",
        "seed": seed,
        "metrics": {
            test10run.ACCURACY_METRIC: accuracy,
            test10run.ECONOMIC_METRIC: economic,
            test10run.SECONDARY_ECONOMIC_METRIC: economic / 40.0,
        },
        "diagnostics": {},
    }


def _arms(economic_gain: float, shuffle_gain: float, accuracy_gain: float) -> list[dict]:
    arms = []
    for offset, seed in enumerate(range(42, 52)):
        jitter = 0.000001 * offset
        arms.append(_arm(test10run.M1_MODEL_ID, seed, 0.380 + jitter, 0.0150 + jitter))
        arms.append(
            _arm(
                test10run.M2_MODEL_ID,
                seed,
                0.380 + jitter + economic_gain,
                0.0150 + jitter + accuracy_gain,
            )
        )
        arms.append(
            _arm(
                test10run.M2_SHUFFLE_MODEL_ID,
                seed,
                0.380 + jitter + economic_gain - shuffle_gain,
                0.0150 + jitter,
            )
        )
    return arms


def _reading(economic_gain: float, shuffle_gain: float, accuracy_gain: float) -> dict:
    arms = _arms(economic_gain, shuffle_gain, accuracy_gain)
    absolute = test10run._absolute_rows(arms)
    _, paired_summary = test10run.paired_tables(
        absolute,
        arms,
        [
            (test10run.M2_MODEL_ID, test10run.M1_MODEL_ID),
            (test10run.M2_SHUFFLE_MODEL_ID, test10run.M1_MODEL_ID),
            (test10run.M2_MODEL_ID, test10run.M2_SHUFFLE_MODEL_ID),
        ],
    )
    return test10run.test10_reading(paired_summary)


def test_reading_confirms_only_with_a_gain_that_needs_the_clv_assignment():
    confirmed = _reading(0.009, 0.008, 0.0)

    assert confirmed["economic_gain_over_m1"] is True
    assert confirmed["accuracy_not_significantly_lower"] is True
    assert confirmed["clv_attribution_over_shuffle"] is True
    assert confirmed["classification"] == "value_basis_confirmed"
    assert confirmed["selection_on_test"] is False


def test_reading_rejects_a_gain_the_shuffle_reproduces():
    shuffle_reproduces = _reading(0.009, 0.0, 0.0)

    assert shuffle_reproduces["economic_gain_over_m1"] is True
    assert shuffle_reproduces["clv_attribution_over_shuffle"] is False
    assert shuffle_reproduces["classification"] == "value_basis_not_confirmed"


def test_reading_rejects_an_economic_gain_bought_with_accuracy():
    traded = _reading(0.009, 0.008, -0.0005)

    assert traded["accuracy_not_significantly_lower"] is False
    assert traded["classification"] == "value_basis_not_confirmed"


def test_reading_rejects_no_economic_gain():
    flat = _reading(0.0, 0.0, 0.0)

    assert flat["economic_gain_over_m1"] is False
    assert flat["classification"] == "value_basis_not_confirmed"


def test_paired_summary_counts_the_seeds_that_moved_the_right_way():
    arms = _arms(0.009, 0.008, 0.0)
    absolute = test10run._absolute_rows(arms)
    _, summary = test10run.paired_tables(
        absolute, arms, [(test10run.M2_MODEL_ID, test10run.M1_MODEL_ID)]
    )
    row = summary[summary.metric.eq(test10run.ECONOMIC_METRIC)].iloc[0]

    assert row["n_seeds"] == 10
    assert row["positive_seed_count"] == 10
    assert np.isclose(row["mean"], 0.009)


def _tiny_model(rho: float):
    indices = torch.tensor([[0, 2, 1, 4], [2, 0, 4, 1]], dtype=torch.long)
    values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adj = torch.sparse_coo_tensor(
        indices, values, (6, 6), check_invariants=False
    ).coalesce()
    torch.manual_seed(5)
    from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN

    return M5NConditionedValueBasisLightGCN(
        n_users=2,
        n_items=4,
        user_q_n=np.array([0.2, 0.8], dtype=np.float32),
        user_q_v=np.array([0.3, 0.7], dtype=np.float32),
        user_q_c=np.array([0.4, 0.9], dtype=np.float32),
        user_clv_valid=np.array([True, True]),
        item_price_percentile=np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float32),
        item_price_valid=np.array([True, True, True, True]),
        adj=adj,
        id_dim=4,
        rho=rho,
        n_layers=1,
        economic_propagation=False,
    )


class _RecordingStore:
    """Minimal ProgressStore stand-in with the real save_epoch signature."""

    def __init__(self):
        self.saved = []

    def restore_epoch(self, model, optimizer, rng):
        return None

    def mark_stage(self, status, **fields):
        return {}

    def heartbeat(self, **fields):
        return True

    def save_epoch(self, model, optimizer, rng, **epoch_state):
        self.saved.append(epoch_state)
        return None


def test_training_loop_runs_every_epoch_and_checkpoints_each_one():
    prepared = {
        "data": {
            # user 0 bought items 0 and 1, user 1 bought item 2; item 3 stays
            # unseen so a uniform negative always exists
            "tr_u": np.array([0, 0, 1, 1], dtype=np.int64),
            "tr_i": np.array([0, 1, 2, 2], dtype=np.int64),
            "pos_key": np.array([0, 1, 6, 6], dtype=np.int64),
            "n_items": 4,
        }
    }
    cfg = test10run.ValueBasisTest10Config(
        out_dir="/tmp/value_basis_test10", epochs=2, batch_size=2
    )
    store = _RecordingStore()

    training = test10run._train_arm(
        _tiny_model(0.25), prepared, cfg, "tiny", 42, store
    )

    assert training["epochs_run"] == 2
    assert training["selection"] == "none"
    assert [state["epoch"] for state in store.saved] == [1, 2]
    assert len(training["history"]) == 2
