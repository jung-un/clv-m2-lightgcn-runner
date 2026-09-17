import numpy as np
import pandas as pd
import pytest
import torch

import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_value_basis_test10 as paired
from test_lightgcn_clv_value_basis_test10 import _RecordingStore, _tiny_model


def _cfg(**overrides):
    return recheck.configure_component_recheck(out_dir="/tmp/component_recheck", **overrides)


def test_protocol_is_five_dev_seeds_single_uniform_negative():
    summary = recheck.preflight_summary(_cfg())

    assert summary["seeds"] == [42, 43, 44, 45, 46]
    assert summary["split"] == "historical_development_days_684_690"
    assert summary["loss"]["negative_count"] == 1
    assert summary["loss"]["hard_negative"] is False
    assert summary["trained_models"] == list(recheck.MODEL_IDS)


def test_config_fails_closed_on_protocol_drift():
    for field, bad in (
        ("seeds", (42,)),
        ("epochs", 50),
        ("negative_count", 5),
        ("history_rho", 0.1),
        ("m4_lambda", 1.0),
        ("m3_beta_cap", 0.5),
    ):
        with pytest.raises(ValueError):
            _cfg(**{field: bad})


def test_each_component_changes_exactly_one_intervention_point():
    specs = {spec["model_id"]: spec for spec in recheck.arm_specifications()}

    def changed(spec):
        return {
            "graph": spec["graph"] != "binary",
            "representation": spec["kind"] != "lightgcn",
            "loss": spec["weighted"],
        }

    assert not any(changed(specs[recheck.M1_MODEL_ID]).values())
    assert changed(specs[recheck.M3_MODEL_ID]) == {"graph": True, "representation": False, "loss": False}
    assert changed(specs[recheck.M2_MODEL_ID]) == {"graph": False, "representation": True, "loss": False}
    assert changed(specs[recheck.M4_MODEL_ID]) == {"graph": False, "representation": False, "loss": True}


def _arms(deltas: dict[str, dict[str, list[float]]]) -> list[dict]:
    metrics = ("recall@10", "ndcg@10", "recall@50", recheck.WEIGHTED_HIT_10, recheck.WEIGHTED_HIT_50)
    arms = []
    for index, seed in enumerate(recheck.SEEDS):
        base = {metric: 0.1 + 0.001 * index for metric in metrics}
        arms.append({"model_id": recheck.M1_MODEL_ID, "role": "baseline", "seed": seed, "metrics": base})
        for model_id in recheck.MODEL_IDS[1:]:
            shifted = {
                metric: value + deltas.get(model_id, {}).get(metric, [0.0] * 5)[index]
                for metric, value in base.items()
            }
            arms.append({"model_id": model_id, "role": "x", "seed": seed, "metrics": shifted})
    return arms


def _reading(deltas):
    cfg = _cfg()
    arms = _arms(deltas)
    absolute = recheck._absolute_rows(arms)
    absolute_summary = pd.DataFrame(
        [{"model_id": model_id, "metric": metric, **recheck.test10._mean_ci(group[metric].to_numpy())}
         for model_id, group in absolute.groupby("model_id", sort=False)
         for metric in paired._metric_columns(absolute)]
    )
    _, paired_summary = paired.paired_tables(
        absolute, arms, [(model_id, recheck.M1_MODEL_ID) for model_id in recheck.MODEL_IDS[1:]]
    )
    return recheck.component_reading(absolute_summary, paired_summary, cfg)


def test_reading_applies_role_specific_rules():
    up4 = [0.01, 0.01, 0.01, 0.01, -0.001]
    up3 = [0.01, 0.01, 0.01, -0.001, -0.001]
    reading = _reading(
        {
            recheck.M3_MODEL_ID: {recheck.WEIGHTED_HIT_10: up4},
            recheck.M2_MODEL_ID: {"recall@50": up4, recheck.WEIGHTED_HIT_50: up3},
            recheck.M4_MODEL_ID: {"ndcg@10": up4, recheck.WEIGHTED_HIT_10: [-0.01] * 5},
        }
    )["components"]

    assert reading[recheck.M3_MODEL_ID]["retained"] is True
    # only 3/5 seeds improved the weighted hit@50
    assert reading[recheck.M2_MODEL_ID]["weighted_hit50_gain"] is False
    assert reading[recheck.M2_MODEL_ID]["retained"] is False
    # top-rank gain, but weighted hit@10 fell below 98% of M1
    assert reading[recheck.M4_MODEL_ID]["top_rank_gain"] is True
    assert reading[recheck.M4_MODEL_ID]["value_guard"] is False


def test_accuracy_guard_rejects_a_value_gain_bought_with_ndcg():
    reading = _reading(
        {recheck.M3_MODEL_ID: {recheck.WEIGHTED_HIT_10: [0.01] * 5, "ndcg@10": [-0.01] * 5}}
    )
    assert reading["components"][recheck.M3_MODEL_ID]["accuracy_guard"] is False
    assert recheck.M3_MODEL_ID not in reading["combination_candidates"]


def test_weighted_loss_path_trains_and_weights_matter():
    prepared = {
        "data": {
            "tr_u": np.array([0, 0, 1, 1], dtype=np.int64),
            "tr_i": np.array([0, 1, 2, 2], dtype=np.int64),
            "pos_key": np.array([0, 1, 6, 6], dtype=np.int64),
            "n_items": 4,
        },
        "m4_weights": np.array([2.0, 0.5, 1.0, 0.5]),
    }
    cfg = recheck.ComponentRecheckConfig(out_dir="/tmp/component_recheck", epochs=2, batch_size=2)
    spec = {"model_id": recheck.M4_MODEL_ID, "weighted": True}
    store = _RecordingStore()

    training = recheck._train_arm(_tiny_model(0.0), prepared, cfg, spec, 42, store)

    assert training["epochs_run"] == 2
    assert [state["epoch"] for state in store.saved] == [1, 2]

    model = _tiny_model(0.0)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[3], [3]])
    plain, _, _ = recheck._batch_loss(model, users, positives, negatives, None)
    weighted, _, _ = recheck._batch_loss(model, users, positives, negatives, torch.tensor([3.0, 0.0]))
    assert not torch.isclose(plain, weighted)


def test_history_fit_arm_uses_its_own_leave_one_out_bpr():
    from test_clv_history_item_fit_model import _model

    model = _model()
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([[1], [0]])

    loss, bpr, correct = recheck._batch_loss(model, users, positives, negatives, None)
    expected, diagnostics = model.bpr_loss(users, positives, negatives[:, 0])

    assert torch.isclose(loss, expected)
    assert bpr == pytest.approx(diagnostics["bpr"])
