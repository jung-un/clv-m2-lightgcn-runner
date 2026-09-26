"""Small CPU check; no external data, GPU training, or baseline fitting."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import clv_shared_nv_feature_screen as screen
from clv_shared_nv_feature_model import SharedNVLightGCN, build_features
from test_linear_nv_original_m4_screen import inputs


def test_shared_l2_is_independent_of_batch_replication():
    torch.manual_seed(43)
    frame = pd.DataFrame(dict(u_idx=[0, 0, 1, 1], i_idx=[0, 1, 1, 2], up=[1., 2., 3., 4.]))
    features = build_features(frame, n_users=2, n_items=5,
        q_n=np.array([.2, .8]), q_v=np.array([.3, .7]), valid=np.ones(2, bool))
    adj = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (7, 7))
    model = SharedNVLightGCN(n_users=2, n_items=5, features=features, adj=adj)
    u, p, n = (torch.tensor(a) for a in ([0, 1], [1, 1], [3, 4]))

    def probe(repeats):
        users, positives, negatives = (a.repeat(repeats) for a in (u, p, n))
        model.zero_grad()
        penalty = model.batch_l2(users, positives, negatives)
        penalty.backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        pos, neg = model._pair_scores(users, positives, negatives)
        return penalty.detach(), gradients, torch.nn.functional.softplus(neg-pos).mean().detach()

    original, duplicated = probe(1), probe(2)
    torch.testing.assert_close(original[2], duplicated[2])  # Mean BPR is unchanged.
    for name in original[1]:
        torch.testing.assert_close(original[1][name], duplicated[1][name], msg=name)
    torch.testing.assert_close(original[0], duplicated[0])
    expected = model.pref_reg * (sum(t.square().sum() for t in
        (model.E_u(u), model.E_i(p), model.E_i(n)))/len(u)
        + sum(w.square().sum() for w in model.encoders.parameters()))
    torch.testing.assert_close(original[0], expected.detach())


def test_shared_features_joint_training_resume_and_readout():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder, patch.object(screen.base.v3, 'DEVICE', 'cpu'):
        cfg = replace(screen.configure(folder), epochs=2, eval_every=1,
                      patience_start=1, patience=1, batch_size=2)
        prep = inputs()
        d = prep['data']
        frame = pd.DataFrame(dict(u_idx=d['tr_u'], i_idx=d['tr_i'], up=[1., 3., 2., 4.]))
        kw = dict(n_users=2, n_items=60, q_n=prep['q_n'], q_v=prep['q_v'], valid=prep['clv_valid'])
        features = build_features(frame, **kw)
        assert not features['loo_n'][0].any() and not features['loo_v'][0].any()
        assert features['loo_n'][1].any() and features['loo_v'][1].any()
        # Item 1 has users 0 and 1. Remove user 0 -> only q_N(1)=.2 + prior.
        from clv_m5_n_conditioned_value_basis_model import fixed_value_basis
        np.testing.assert_allclose(features['loo_n'][1], fixed_value_basis(
            np.array([(.2+10*.5)/11]), np.array([True]))[0])
        changed = build_features(frame, **dict(kw, q_n=np.array([.99, .2])))
        np.testing.assert_allclose(changed['loo_n'][1], features['loo_n'][1])
        assert not np.allclose(changed['user_n'], features['user_n'])
        assert build_features(frame, **dict(kw, q_n=np.array([0., .2])))['user_n'][0].any()
        invalid = build_features(frame, **dict(kw, q_n=np.array([np.nan, .2]), valid=np.array([False, True])))
        assert not invalid['user_n'][0].any() and not invalid['user_v'][0].any()
        prep.update(features=features, features_sha256=screen.feature_hash(features),
            feature_settings=dict(alpha_n=.05, alpha_v=.05, shrinkage=10., bandwidth=.25),
            screen_config=asdict(cfg), source_report='synthetic', source_report_sha256='synthetic')
        prep['feature_settings_sha'] = screen.base._digest(prep['feature_settings'])
        _, prep['m4_weights'], prep['m4_diagnostics'] = screen.prior.lambda025.previous.audit_weights(prep, cfg)
        model = screen._build(prep, cfg)
        baseline = screen.base._build_model(prep, screen.es.strength_cfg(cfg),
            dict(kind='id', graph='binary', rho=0.), 43)
        assert torch.equal(model.E_u.weight, baseline.E_u.weight)
        assert torch.equal(model.E_i.weight, baseline.E_i.weight)
        assert sum(p.numel() for p in model.encoders.parameters()) == 48
        u, p, n = (torch.tensor(a) for a in ([0, 1], [1, 1], [3, 4]))
        model.pref_reg = 0.  # Prove N/V gradient comes from BPR, not L2.
        loss, _ = model.bpr_loss(u, p, n)
        loss.backward()
        assert all(layer.weight.grad.norm() > 0 for layer in model.encoders.values())
        assert model.E_u.weight.grad.norm() > 0 and model.E_i.weight.grad.norm() > 0
        uid, iid, *_ = model.embeddings()
        _, negative = model._pair_scores(u, p, n)
        torch.testing.assert_close(negative, (uid[u]*iid[n]).sum(1))
        weights = torch.tensor([.7, 1.3])
        pos, neg = model._pair_scores(u, p, n)
        weighted, _ = model.weighted_bpr_loss(u, p, n, weights)
        torch.testing.assert_close(weighted, (weights*torch.nn.functional.softplus(neg-pos)).mean())
        split = screen.base.hm_budget._score_share(model, prep)
        assert 'n_top10_score_mean_abs' in split and 'v_top10_score_mean_abs' in split
        assert set(s['model_id'] for s in screen.specs()) == {screen.M2, screen.M5}
        metrics = {k: 1. for k in (*screen.base.ACCURACY, *screen.es.fixed.PRIMARY)}
        metrics['고CLV_revenue@10'] = .5
        anchors = [dict(model_id=mid, seed=43, origin='reused', metrics=metrics,
            selected_epoch=1, stopped_epoch=2, curve=[dict(epoch=1, metrics=metrics)])
            for mid in ('m1', screen.M4)]
        original_save = screen.ProgressStore.save_epoch
        def interrupt_after_saved_epoch(store, *args, **kwargs):
            original_save(store, *args, **kwargs)
            raise InterruptedError('simulated runtime interruption')
        with patch.object(screen, 'configure', return_value=cfg), \
             patch.object(screen.prior, 'verified_anchors', return_value=anchors), \
             patch.object(screen.base.capacity, '_evaluate', return_value=metrics):
            with patch.object(screen.ProgressStore, 'save_epoch', interrupt_after_saved_epoch):
                try:
                    screen.run(cfg, prep)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError('Interruption check not exercised')
            paths = screen.run(cfg, prep)
            report = json.loads(Path(paths['json']).read_text())
            assert report['reading']['complete'] and not report['final_test'] and not report['holdout']
            assert report['code_version'] == screen.VERSION
            assert report['regularization']['shared_nv_coefficient'] == cfg.pref_reg
            assert report['arms'][2]['diagnostics']['shared_nv_l2_coefficient'] == cfg.pref_reg
            assert report['arms'][2]['training']['resumed_from_epoch'] == 1
            assert report['arms'][3]['training']['resumed_from_epoch'] == 0
            assert not report['reading'][screen.M5]['both_economic_at10_above_m4']
            assert '고CLV_revenue@10' in pd.read_csv(paths['absolute']).columns
            assert len(report['arms']) == 4
            # Completed arms must reuse; no training or optimizer after cache hit.
            with patch.object(screen.es, '_train', side_effect=AssertionError('unexpected training')):
                screen.run(cfg, prep)
        with patch.object(screen.base, '_prepare', side_effect=AssertionError('unexpected data load')):
            # An existing v1 report cannot be overwritten by the correction.
            report['code_version'] = 'shared-nv-feature-m2-m5-seed43-development-v1'
            Path(paths['json']).write_text(json.dumps(report))
            try:
                screen.prepare(Path(folder)/'missing.json', folder)
            except ValueError as exc:
                assert 'another experiment version' in str(exc)
            else:
                raise AssertionError('Old output directory accepted')
            try:
                screen.prepare(Path(folder)/'missing.json', Path(folder)/'unused')
            except ValueError:
                pass
            else:
                raise AssertionError('Missing anchors accepted')


if __name__ == '__main__':
    test_shared_l2_is_independent_of_batch_replication()
    test_shared_features_joint_training_resume_and_readout()
    print('Shared N/V CPU smoke passed')
