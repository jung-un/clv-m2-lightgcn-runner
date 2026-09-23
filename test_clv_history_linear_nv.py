from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from clv_history_item_fit_model import build_personal_history_weights, HistoryItemFitLightGCN
from clv_history_linear_nv_model import LinearNVHistoryLightGCN
from test_clv_history_item_fit_model import _adj, _history_frame
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_history_linear_nv as runner


def make_model(constant=False, q=(.8, .2), valid=(True, True), frame=None, cls=LinearNVHistoryLightGCN):
    torch.manual_seed(43)
    kwargs = dict(n_users=2, n_items=3,
        history=build_personal_history_weights(_history_frame() if frame is None else frame, n_users=2, n_items=3),
        q_n=np.array(q), q_v=np.array([.3, .9]), activity_valid=np.array(valid), value_valid=np.array(valid),
        adj=_adj(), id_dim=6, axis_dim=4, n_layers=1, rho=.05, pref_reg=.001)
    return cls(**kwargs, **({"constant_q": constant} if cls is LinearNVHistoryLightGCN else {}))


def test_parameter_count_and_base_initialization_unchanged():
    old = make_model(cls=HistoryItemFitLightGCN)
    new = make_model()
    assert sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()) == 48
    for key, value in old.named_parameters():
        assert torch.equal(value, dict(new.named_parameters())[key])
    assert new.total_dim == 14


def test_affine_raw_profile_formula_and_constant_control():
    m = make_model()
    source, _, _, _ = m._axis_tables()
    p = torch.sparse.mm(m.activity_history, source)
    expected = m.n_transform(torch.cat([p, m.q_n[:, None]], 1))
    assert torch.allclose(m._full_history_profiles()[0], expected)
    other = make_model(q=(.1, .9))
    assert not torch.allclose(m._full_history_profiles()[0], other._full_history_profiles()[0])
    a, b = make_model(True), make_model(True, q=(.1, .9))
    assert torch.equal(a._full_history_profiles()[0], b._full_history_profiles()[0])
    assert torch.equal(a.n_transform.weight, m.n_transform.weight)


def test_invalid_and_empty_loo_are_zero_even_with_bias():
    frame = _history_frame()
    frame = frame[~((frame.u_idx == 1) & (frame.i_idx == 1))]
    m = make_model(valid=(False, True), frame=frame)
    with torch.no_grad():
        m.n_transform.bias.fill_(2)
        m.v_transform.bias.fill_(2)
    assert m._full_history_profiles()[0][0].count_nonzero() == 0
    n, _, v, _ = m._training_profiles(torch.tensor([1]), torch.tensor([2]))
    assert n.count_nonzero() == v.count_nonzero() == 0


def test_loo_removes_positive_source_from_profile():
    m = make_model()
    n, _, _, _ = m._training_profiles(torch.tensor([0]), torch.tensor([0]))
    source, _, _, _ = m._axis_tables()
    expected = m.n_transform(torch.cat([source[1:2], m.q_n[:1, None]], 1))
    assert torch.allclose(n, expected, atol=1e-6)
    n.sum().backward()
    assert m.activity_source.weight.grad[0].abs().max() < 1e-6
    assert m.activity_source.weight.grad[1].norm() > 0


@pytest.mark.parametrize("weighted", [False, True])
def test_joint_recommendation_gradients_without_regularization(weighted):
    m = make_model()
    m.pref_reg = 0
    u, p, n = torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([2, 0])
    loss, _ = m.weighted_bpr_loss(u, p, n, torch.tensor([1.2, .8])) if weighted else m.bpr_loss(u, p, n)
    loss.backward()
    for name, param in m.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.norm() > 0, name


def test_l2_extends_existing_penalty_without_batch_size_dilution():
    m = make_model()
    u, p, n = torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([2, 0])
    expected = HistoryItemFitLightGCN.batch_l2(m, u, p, n) + m.pref_reg * sum(
        x.square().sum() for layer in (m.n_transform, m.v_transform) for x in layer.parameters())
    assert torch.equal(m.batch_l2(u, p, n), expected)


def test_bad_q_rejected():
    with pytest.raises(ValueError, match="finite"):
        make_model(q=(np.nan, .2))


def test_protocol_locks_and_cost(tmp_path):
    cfg = runner.configure(out_dir=str(tmp_path))
    assert runner.preflight(cfg)["new_fits"] == 12
    for overrides in ({"m4_mode": "original"}, {"dataset": "hm"}, {"rhos": (.1,)}, {"epochs": 300}):
        with pytest.raises(ValueError):
            runner.configure(out_dir=str(tmp_path), **overrides)


def test_missing_anchor_stops_before_training(tmp_path, monkeypatch):
    cfg = runner.configure(out_dir=str(tmp_path), seeds=(43,), reuse_dirs=(str(tmp_path),))
    monkeypatch.setattr(runner.base, "_prepare", lambda cfg: {"input_hash": "synthetic"})
    monkeypatch.setattr(runner.base, "_reuse_component", lambda *args: None)
    def forbidden(*args, **kwargs):
        pytest.fail("Training must never run without compatible anchors")
    monkeypatch.setattr(runner, "_run_new", forbidden)
    with pytest.raises(RuntimeError, match="No training started"):
        runner.run(cfg)


def test_existing_strength_identity_is_reused(tmp_path, monkeypatch):
    cfg = runner.configure(out_dir=str(tmp_path / "new"), seeds=(43,), reuse_dirs=(str(tmp_path),))
    prepared = {"input_hash": "synthetic"}
    for spec in runner.base.arm_specifications(cfg):
        identity = runner.base._identity(prepared, cfg, spec, 43)
        path = tmp_path / "arms" / runner.base._digest(identity) / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(**spec, seed=43, identity=identity,
            metrics={m: 1. for m in (*runner.base.ACCURACY, *runner.PRIMARY)})))
    anchors = runner.load_anchors(prepared, cfg)
    assert len(anchors) == 4 and all(a["origin"] == "reused_strength" for a in anchors)


def test_overall_not_highclv_is_primary_and_missing_seed_not_pass():
    cfg = runner.configure(seeds=(42,43))
    arms = []
    for spec in runner.base.arm_specifications(cfg) + runner.new_specs():
        metrics = {m: 1. for m in (*runner.base.ACCURACY, *runner.PRIMARY)}
        if "linear" in spec["model_id"]:
            metrics.update({m: 1.01 for m in runner.PRIMARY})
        metrics["고CLV_revenue@10"] = .1 if "linear" in spec["model_id"] else 1.
        arms.append(dict(**spec, seed=42, origin="synthetic", metrics=metrics))
    _, comparison, _, reading = runner.tables(arms, cfg.seeds)
    assert reading[reading.seed == 42].directional_pass.all()
    assert not reading[reading.seed == "mean"].complete.any()
    assert ((comparison.model_id == "m5_linear_nv") & (comparison.reference == "m5_linear_constant")).any()
    arms[-1]["metrics"][runner.PRIMARY[0]] = float("nan")
    assert not runner.tables(arms, cfg.seeds)[-1].query("model_id == 'm5_linear_constant'").complete.any()


@pytest.mark.parametrize("weighted", [False, True])
def test_real_training_loop_resume_matches_uninterrupted(tmp_path, monkeypatch, weighted):
    monkeypatch.setattr(runner.base.v3, "DEVICE", torch.device("cpu"))
    torch.set_num_threads(1)
    prepared = {"data": {"tr_u": np.array([0,0,1,1]), "tr_i": np.array([0,1,1,2]),
        "pos_key": np.array([0,1,4,5]), "n_users": 2, "n_items": 3},
        "m4_weights": np.array([1.2,.8,1.,1.])}
    cfg = replace(runner.configure(), epochs=2, batch_size=2)
    spec = dict(model_id="synthetic", weighted=weighted)
    def store(name):
        return ProgressStore(tmp_path / name, RunIdentity(stage="test", model_id="synthetic", seed=43,
            config_hash="test", source_revision="test", input_hash="test"))
    full, staged = make_model(), make_model()
    runner.base.components._train_arm(full, prepared, cfg, spec, 43, store("full"))
    runner.base.components._train_arm(staged, prepared, replace(cfg, epochs=1), spec, 43, store("resume"))
    restored = make_model()
    runner.base.components._train_arm(restored, prepared, cfg, spec, 43, store("resume"))
    for (_, a), (_, b) in zip(full.named_parameters(), restored.named_parameters()):
        assert torch.equal(a, b)
    assert torch.equal(full.embeddings()[0], restored.embeddings()[0])


def test_all_four_new_arms_save_diagnostics_and_cache(tmp_path, monkeypatch):
    from test_lightgcn_clv_history_m5_strength import prepared
    monkeypatch.setattr(runner.base.v3, "DEVICE", "cpu")
    cfg = replace(runner.configure(out_dir=str(tmp_path), seeds=(43,)), epochs=2)
    prep = prepared()
    # Ranking metrics are a stub here; real score-share, training and disk cache run.
    monkeypatch.setattr(runner.base.capacity, "_evaluate", lambda model, prep:
        {m: .01 for m in (*runner.base.ACCURACY, *runner.PRIMARY)})
    arms = []
    for spec in runner.new_specs():
        p = runner._run_new(prep, cfg, spec, 43)
        assert p["diagnostics"]["affine_parameter_count"] == 48
        assert np.isfinite(p["diagnostics"]["clv_score_share"])
        assert runner._run_new(prep, cfg, spec, 43)["identity"] == p["identity"]
        arms.append(p)
    # Report writer requires the real registered 100-epoch protocol, no training here.
    for spec in runner.base.arm_specifications(cfg):
        arms.append(dict(**spec, seed=43, origin="synthetic", metrics=arms[0]["metrics"]))
    saved = runner.save(arms, replace(cfg, epochs=100))
    assert all(__import__("pathlib").Path(p).is_file() for p in saved.attrs["paths"].values())
