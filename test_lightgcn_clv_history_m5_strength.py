from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import lightgcn_clv_history_m5_strength as runner
from clv_run_state import ProgressStore, RunIdentity


def config(tmp_path, **kwargs):
    return runner.configure_strength(m4_mode="complementary", out_dir=str(tmp_path), **kwargs)


def prepared():
    frame = pd.DataFrame({"u_idx": [0, 0, 1, 1], "i_idx": [0, 1, 1, 2],
                          "b_raw": [1, 2, 3, 4], "v": [1., 3., 2., 4.]})
    users, items = frame.u_idx.to_numpy(), frame.i_idx.to_numpy()
    n_items = 60
    adj = runner.v3.build_adj(users, items, np.ones(4, np.float32), 2, n_items)
    return {"data": {"n_users": 2, "n_items": n_items, "tr_u": users, "tr_i": items,
                     "pos_key": np.sort(users*n_items+items), "adj": adj,
                     "csr_ptr": np.array([0, 2, 4]), "csr_items": items},
            "history": runner.build_personal_history_weights(frame, n_users=2, n_items=n_items),
            "q_n": np.array([.8, .2], np.float32), "q_v": np.array([.3, .9], np.float32),
            "q_c": np.array([.5, .8], np.float32), "clv_valid": np.ones(2, bool),
            "item_amount_percentile": np.linspace(0, 1, n_items, dtype=np.float32),
            "item_economic_valid": np.ones(n_items, bool), "m4_weights": np.array([.7, 1.3, .8, 1.2]),
            "m4_diagnostics": {"row_weight_cv": .2}, "cache": SimpleNamespace(users=np.array([0, 1])),
            "input_hash": "synthetic", "revision": "test"}


def test_scope_and_shared_baselines(tmp_path):
    cfg = config(tmp_path)
    arms = runner.arm_specifications(cfg)
    assert len(arms) == 8
    assert len([a for a in arms if a["role"] == "M1"]) == 1
    assert len([a for a in arms if a["role"] == "M4"]) == 1
    for rho in cfg.rhos:
        pair = [a for a in arms if a["rho"] == rho]
        assert {a["role"] for a in pair} == {"M2", "M5"}
    with pytest.raises(ValueError):
        runner.configure_strength()
    with pytest.raises(ValueError):
        config(tmp_path, epochs=300)
    assert runner.configure_strength(dataset="hm", m4_mode="original").batch_size == 131072


def test_weighted_loss_is_used_and_both_axes_receive_gradient(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.v3, "DEVICE", "cpu")
    prep, cfg = prepared(), config(tmp_path)
    spec = next(a for a in runner.arm_specifications(cfg) if a["role"] == "M5")
    model = runner._build_model(prep, cfg, spec, 43)
    users = torch.tensor([0, 0, 1, 1])
    positives = torch.tensor([0, 1, 1, 2])
    negatives = torch.tensor([[3], [4], [3], [4]])
    weights = torch.tensor(prep["m4_weights"], dtype=torch.float32)
    loss, _, _ = runner.components._batch_loss(model, users, positives, negatives, weights)
    pos, neg = model._pair_scores(users, positives, negatives[:, 0])
    expected = (weights*torch.nn.functional.softplus(neg-pos)).mean()+model.batch_l2(users, positives, negatives[:, 0])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for name in ("activity_source", "activity_target", "value_source", "value_target", "E_u", "E_i"):
        assert getattr(model, name).weight.grad.norm() > 0


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.v3, "DEVICE", "cpu")
    prep = prepared()
    cfg = replace(config(tmp_path), epochs=3, batch_size=2)
    spec = next(a for a in runner.arm_specifications(cfg) if a["role"] == "M5")
    identity = RunIdentity("test", spec["model_id"], 43, "same", "same", "synthetic")
    full = runner._build_model(prep, cfg, spec, 43)
    runner.components._train_arm(full, prep, cfg, spec, 43, ProgressStore(tmp_path/"full", identity))
    partial = runner._build_model(prep, cfg, spec, 43)
    store = ProgressStore(tmp_path/"resume", identity)
    runner.components._train_arm(partial, prep, replace(cfg, epochs=1), spec, 43, store)
    resumed = runner._build_model(prep, cfg, spec, 43)
    training = runner.components._train_arm(resumed, prep, cfg, spec, 43, store)
    assert training["resumed_from_epoch"] == 1
    for name, value in full.state_dict().items():
        if value.is_sparse:
            torch.testing.assert_close(value.to_dense(), resumed.state_dict()[name].to_dense())
        else:
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_reporting_does_not_call_positive_mean_a_seed_consistent_pass(tmp_path):
    cfg = config(tmp_path)
    arms = []
    for seed in cfg.seeds:
        for spec in runner.arm_specifications(replace(cfg, rhos=(.05,))):
            metrics = {k: 1.0 for k in (*runner.ACCURACY, runner.PRIMARY)}
            if spec["role"] == "M5":
                metrics[runner.PRIMARY] = 1.3 if seed == 44 else .95
                metrics["recall@20"] = .98 if seed == 42 else 1.0
            arms.append({**spec, "seed": seed, "origin": "synthetic", "metrics": metrics})
    absolute, comparison, summary, decisions = runner.result_tables(arms, cfg.seeds)
    assert decisions[decisions.seed.eq("mean")].directional_pass.item()
    assert decisions[decisions.seed.eq(42)].directional_pass.item() is False
    assert summary[(summary.reference == "m4") & (summary.metric == runner.PRIMARY)].positive_seeds.item() == 1
    partial = runner.result_tables(arms[:-4], cfg.seeds)[-1]
    assert not partial[partial.seed.eq("mean")].complete.item()
    saved = runner.save_report(arms, cfg)
    assert not any(isinstance(v, pd.DataFrame) for v in saved.attrs.values())


def test_every_arm_trains_evaluates_and_reuses_its_completed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.v3, "DEVICE", "cpu")
    prep, cfg = prepared(), replace(config(tmp_path), epochs=2)
    monkeypatch.setattr(runner.capacity, "_evaluate", lambda model, prepared: {"recall@10": .01})
    for spec in runner.arm_specifications(cfg):
        payload = runner._run_arm(prep, cfg, spec, 43)
        assert payload["training"]["epochs_run"] == 2
        assert payload["metrics"]["recall@10"] == .01
        assert runner._run_arm(prep, cfg, spec, 43)["identity"] == payload["identity"]


def test_hash_separates_dataset_and_m4_but_not_seed_subset(tmp_path):
    cfg, prep = config(tmp_path), prepared()
    spec = runner.arm_specifications(cfg)[1]
    identity = runner._identity(prep, cfg, spec, 42)
    assert identity == runner._identity(prep, replace(cfg, seeds=(42,)), spec, 42)
    assert identity != runner._identity(prep, replace(cfg, m4_mode="original"), spec, 42)
    assert identity != runner._identity(prep, replace(cfg, dataset="hm"), spec, 42)


def test_component_reuse_rejects_other_input_and_m4_formula(tmp_path):
    import json
    prep, cfg = prepared(), config(tmp_path)
    cfg = replace(cfg, reuse_dirs=(str(tmp_path),))
    spec = runner.arm_specifications(cfg)[1]
    old_id = runner.components.M4_MODEL_ID
    cp = tmp_path/"arms"/"old"/f"{old_id}_s43.pt"
    cp.parent.mkdir(parents=True)
    old_cfg = runner.asdict(runner.components.configure_component_recheck())
    def save(input_hash):
        torch.save({"config": old_cfg, "seed": 43, "model_id": old_id,
                    "input_hash": input_hash}, cp)
        payload = {"checkpoint": str(cp), "checkpoint_sha256": runner.file_sha256(cp),
                   "code_version": runner.components.CODE_VERSION, "final_epoch": 100,
                   "split": runner.split_name(cfg), "metrics": {"recall@10": .01}}
        cp.with_suffix(".json").write_text(json.dumps(payload))
    save("other-input")
    assert runner._reuse_component(prep, cfg, spec, 43) is None
    save("synthetic")
    assert runner._reuse_component(prep, cfg, spec, 43)["origin"] == "reused_component"
    assert runner._reuse_component(prep, replace(cfg, m4_mode="original"), spec, 43) is None
