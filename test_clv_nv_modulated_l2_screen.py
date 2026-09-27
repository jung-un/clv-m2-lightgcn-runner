"""Bounded CPU check: one factor/one arm, resume, complete cache and full readout."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import torch

import clv_nv_modulated_l2_screen as screen
from test_lightgcn_clv_history_m5_strength import prepared


def test_one_m2_l2_only_resume_and_cache():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder, patch.object(screen.base.v3, 'DEVICE', 'cpu'):
        registered = screen.configure(folder)
        assert registered.seeds == (43,) and registered.epochs == 300
        assert registered.pref_reg == .001 and registered.negative_count == 1
        assert registered.id_dim == 64 and registered.n_layers == 2
        assert screen.spec()['role'] == 'M2' and not screen.spec()['weighted']
        assert {k for k in screen.SETTINGS if screen.SETTINGS[k] != screen.OLD_SETTINGS[k]} == {'shared_l2'}
        with patch.object(screen.base, '_prepare', side_effect=AssertionError('must not load data')):
            bad = Path(folder)/'bad.json'
            bad.write_text('{}')
            try:
                screen.prepare(bad, Path(folder)/'unused')
            except ValueError as exc:
                assert 'Exact completed' in str(exc)
            else:
                raise AssertionError('Mismatched source was accepted')
        cfg = replace(registered, epochs=2, eval_every=1, patience_start=1, patience=1, batch_size=2)
        prep = prepared()
        d = prep['data']
        frame = pd.DataFrame(dict(u_idx=d['tr_u'], i_idx=d['tr_i'], up=[1., 3., 2., 4.]))
        features = screen.previous.build_features(frame, n_users=2, n_items=d['n_items'],
            q_n=prep['q_n'], q_v=prep['q_v'], valid=prep['clv_valid'])
        prep.update(features=features, features_sha256=screen.previous.feature_hash(features),
            feature_settings=dict(screen.SETTINGS), feature_settings_sha=screen.base._digest(screen.SETTINGS),
            screen_config=asdict(cfg), source_report='synthetic', source_report_sha256=screen.SOURCE_SHA)
        new = screen.previous._build(prep, cfg)
        old = screen.previous._build(dict(prep, feature_settings=screen.OLD_SETTINGS), cfg)
        assert new.shared_l2 == .0001 and old.shared_l2 == .001
        for left, right in zip(new.parameters(), old.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        # Same scores and ID regularization; only nonzero N/V L2 gradients change tenfold.
        with torch.no_grad():
            for model in (new, old):
                for layer in model.encoders.values():
                    layer.weight.add_(.01)
        u, p, n = (torch.tensor(a) for a in ([0, 1], [1, 1], [3, 4]))
        for left, right in zip(new._pair_scores(u, p, n), old._pair_scores(u, p, n)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        new.batch_l2(u, p, n).backward(); old.batch_l2(u, p, n).backward()
        torch.testing.assert_close(new.E_u.weight.grad, old.E_u.weight.grad)
        torch.testing.assert_close(new.E_i.weight.grad, old.E_i.weight.grad)
        for name in new.encoders:
            torch.testing.assert_close(new.encoders[name].weight.grad, .1*old.encoders[name].weight.grad)
        metrics = {k: 1. for k in (*screen.base.ACCURACY, *screen.es.fixed.PRIMARY)}
        metrics.update({'저CLV_revenue@10': .5, '중CLV_revenue@10': .6, '고CLV_revenue@10': .7,
                        'coverage@10': .2})
        anchors = [dict(model_id=mid, seed=43, origin='synthetic', metrics=metrics,
            selected_epoch=1, stopped_epoch=2,
            curve=[dict(epoch=e, metrics=metrics) for e in (1, 2)]) for mid in ('m1', screen.previous.M2)]
        original = screen.ProgressStore.save_epoch
        def interrupt(store, *args, **kwargs):
            original(store, *args, **kwargs)
            raise InterruptedError('simulated disconnect after complete epoch')
        with patch.object(screen, 'configure', return_value=cfg), \
             patch.object(screen, 'read_source', return_value=({}, anchors)), \
             patch.object(screen.base.capacity, '_evaluate', return_value=metrics):
            with patch.object(screen.ProgressStore, 'save_epoch', interrupt):
                try:
                    screen.run(cfg, prep)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError('Resume path not exercised')
            paths = screen.run(cfg, prep)
            report = json.loads(Path(paths['json']).read_text())
            assert len(report['arms']) == 3 and report['new_fit_count'] == 1
            assert [a['model_id'] for a in report['new_arms']] == [screen.MODEL_ID]
            assert not report['final_test'] and not report['holdout']
            arm = report['arms'][-1]
            assert arm['training']['resumed_from_epoch'] == 1
            assert arm['diagnostics']['shared_nv_l2_coefficient'] == .0001
            assert arm['diagnostics']['nv_activity_ok']
            assert not report['reading'][screen.MODEL_ID]['both_economic_at10_above_m1']
            assert '고CLV_revenue@10' in pd.read_csv(paths['absolute']).columns
            assert len(pd.read_csv(paths['comparison'])) == 3*len(metrics)
            assert len(pd.read_csv(paths['same_epoch_comparison'])) == 6*len(metrics)
            with patch.object(screen.es, '_train', side_effect=AssertionError('must reuse complete arm')):
                screen.run(cfg, prep)
        assert len(list(Path(folder).glob('arms/*/result.json'))) == 1


if __name__ == '__main__':
    test_one_m2_l2_only_resume_and_cache()
    print('One-arm N/V L2 CPU smoke passed')
