"""Bounded CPU check: equations, masks, true evaluator, interruption/resume/cache."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import clv_shared_feature_residual_m2_screen as s
from clv_shared_feature_residual_m2 import AXES, centered_basis, build_features
from test_lightgcn_clv_history_m5_strength import prepared


def test_shared_feature_residual_screen():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder, patch.object(s.base.v3, 'DEVICE', 'cpu'):
        cfg0 = s.configure(folder)
        assert (cfg0.seeds, cfg0.epochs, cfg0.id_dim, cfg0.n_layers) == ((43,), 300, 64, 2)
        assert cfg0.negative_count == 1 and cfg0.pref_reg == s.SETTINGS['user_l2'] == .001
        with patch.object(s.base, '_prepare', side_effect=AssertionError('must not prepare data')):
            bad = Path(folder)/'bad.json'
            bad.write_text('{}')
            try:
                s.prepare(bad, folder)
            except ValueError as exc:
                assert 'Exact completed' in str(exc)
            else:
                raise AssertionError('Invalid reference accepted')
        x, _ = centered_basis([0., .5, np.nan, 1.], [True, True, False, True])
        np.testing.assert_allclose(x[[0, 1, 3]].mean(0), 0, atol=3e-8)
        assert np.any(x[0]) and not np.any(x[2])
        tied, _ = centered_basis([0., 0.], [True, True])
        assert not np.any(tied)  # No artificial jitter/variance amplification.
        cfg = replace(cfg0, epochs=2, eval_every=1, patience_start=1, patience=1, batch_size=2)
        prep = prepared()
        d = prep['data']
        frame = pd.DataFrame(dict(u_idx=d['tr_u'], i_idx=d['tr_i'],
                                 up=[1., 3., 2., 4.], cat_idx=[0, 0, 0, 0]))
        features = build_features(frame, n_users=2, n_items=d['n_items'], q_n=prep['q_n'],
            q_v=prep['q_v'], n_valid=[True, True], v_valid=[True, True])
        assert not features['item_price'][3:].any() and not features['item_within'][3:].any()
        features['keys'] = d['pos_key']
        prep.update(features=features, features_sha256=s.feature_hash(features),
            feature_settings=json.loads(json.dumps(s.SETTINGS)), screen_config=asdict(cfg),
            source_report='synthetic', source_report_sha256=s.source.SOURCE_SHA)
        model = s.build(prep, cfg)
        m1cfg = s.base.capacity.CapacitySearchConfig()
        m1spec = dict(model_id=s.base.capacity.M1_MODEL_ID, id_dim=64, pref_reg=.001)
        m1 = s.base.capacity._build_model(prep, m1cfg, m1spec, 43)
        actual_u, actual_i = model.embeddings()[:2]
        expected_u, expected_i = m1.embeddings()[:2]
        torch.testing.assert_close(actual_u @ actual_i.T, expected_u @ expected_i.T,
                                   rtol=1e-5, atol=2e-6)
        assert sum(p.numel() for p in model.encoders.parameters()) == 768
        assert all(p.requires_grad for p in model.parameters())
        assert s.diagnose(model, prep, cfg, s.spec(), 0)['nv_activity_ok']
        u, p, n = (torch.tensor(a) for a in ([0, 1], [1, 2], [3, 4]))
        loss, _ = model.bpr_loss(u, p, n)
        loss.backward()
        assert all(model.encoders[a].weight.grad.norm() > 0 for a in AXES)
        with torch.no_grad():
            for layer in model.encoders.values():
                layer.weight.add_(.01)
        # Raw ID, not effective layer0; four shared penalties are not /B.
        expected = .001*((model.E_u(u).square().sum()+model.E_i(p).square().sum()
            +model.E_i(n).square().sum())/len(u)+sum(w.square().sum() for w in model.encoders.parameters()))
        torch.testing.assert_close(model.batch_l2(u, p, n), expected)
        torch.testing.assert_close(model.batch_l2(u.repeat(3), p.repeat(3), n.repeat(3)), expected)
        full_user = model.embeddings()[0]
        for name in AXES:
            model.disabled_axes = (name,)
            assert not torch.equal(model.embeddings()[0], full_user)
        model.disabled_axes = ()
        # Real full-catalog evaluator and new-to-user masking, not mocked metrics.
        prep['base_cfg'] = dict(s.base.v3.CFG, EVAL_BATCH=2, KS=[10, 20, 50])
        gt, amounts = {0: np.array([3, 4]), 1: np.array([4, 5])}, {0: np.array([2., 1.]), 1: np.array([1., 3.])}
        prep['cache'] = s.base.v3.EvalCache(gt, amounts, np.array([1., 3.]), (1.5, 2.5), d['n_items'])
        prep['meta'] = dict(price_pct=np.linspace(0, 1, d['n_items']),
            pop_prob=np.ones(d['n_items'])/d['n_items'], cat=np.arange(d['n_items']) % 3)
        metrics = s.base.capacity._evaluate(m1, prep)
        assert len(metrics) > 100 and '고CLV_revenue@10' in metrics
        anchors = [dict(model_id='m1', seed=43, origin='synthetic', metrics=metrics,
            selected_epoch=1, stopped_epoch=2, curve=[dict(epoch=e, metrics=metrics) for e in (1, 2)])]
        original = s.ProgressStore.save_epoch
        def interrupt(store, *args, **kwargs):
            original(store, *args, **kwargs)
            raise InterruptedError('simulated disconnect after saved epoch')
        with patch.object(s, 'configure', return_value=cfg), \
             patch.object(s, 'read_source', return_value=({}, anchors)):
            with patch.object(s.ProgressStore, 'save_epoch', interrupt):
                try:
                    s.run(cfg, prep)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError('Interruption path not exercised')
            paths = s.run(cfg, prep)
            report = json.loads(Path(paths['json']).read_text())
            arm = report['arms'][-1]
            assert len(report['arms']) == 2 and report['new_fit_count'] == 1
            assert arm['training']['resumed_from_epoch'] == 1 and arm['diagnostics']['nv_activity_ok']
            assert not report['final_test'] and not report['holdout']
            assert len(pd.read_csv(paths['comparison'])) == len(metrics)
            assert len(pd.read_csv(paths['same_epoch_comparison'])) == 2*len(metrics)
            tops = pd.read_csv(paths['recommendations'])
            assert len(tops) == 4*2*50
            assert not any(u*d['n_items']+i in set(d['pos_key']) for u, i in zip(tops.user, tops.item))
            with patch.object(s.es, '_train', side_effect=AssertionError('must reuse completed arm')):
                s.run(cfg, prep)
            # Same final weights as uninterrupted training, including optimizer/RNG restore.
            full_dir = str(Path(folder)/'uninterrupted')
            full_cfg = replace(cfg, out_dir=full_dir)
            full_prep = dict(prep, screen_config=asdict(full_cfg))
            with patch.object(s, 'configure', return_value=full_cfg):
                full_paths = s.run(full_cfg, full_prep)
            full_arm = json.loads(Path(full_paths['json']).read_text())['arms'][-1]
            left = torch.load(arm['checkpoint'], weights_only=False)['model_state']
            right = torch.load(full_arm['checkpoint'], weights_only=False)['model_state']
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


if __name__ == '__main__':
    test_shared_feature_residual_screen()
    print('Shared feature residual M2 CPU check passed')
