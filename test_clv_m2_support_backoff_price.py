"""Minimal checks for one support-adaptive item-price coordinate with user N/V."""

from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict, replace
from unittest.mock import patch
import json

import numpy as np
import pandas as pd
import torch

import clv_m2_support_backoff_price_screen as screen
import clv_m2_support_backoff_price as support_model
from clv_m2_support_backoff_price import (
    PRICE_AXIS,
    SupportBackoffPriceLightGCN,
    build_support_backoff_features,
    support_backoff_price,
)
from lightgcn_clv_v3 import EvalCache, build_adj
from test_lightgcn_clv_history_m5_strength import prepared


def test_support_backoff_price_prefers_fine_and_shrinks_sparse_items():
    values, valid, reliability, source = support_backoff_price(
        global_price=np.array([.9, .8, .7, .6]),
        coarse_price=np.array([.7, .6, .5, .4]),
        fine_price=np.array([.2, .3, .4, np.nan]),
        fine_valid=np.array([True, True, True, False]),
        coarse_valid=np.array([True, True, True, True]),
        observation_count=np.array([1, 10, 100, 10]),
        shrinkage=10.0,
    )
    np.testing.assert_allclose(reliability, [1 / 11, .5, 10 / 11, .5])
    np.testing.assert_allclose(values[:3], .5 + reliability[:3] * (np.array([.2, .3, .4]) - .5))
    assert values[0] > .2 and values[2] > .4
    assert abs(values[0] - .5) < abs(values[2] - .5)  # Sparse support shrinks more.
    assert values[3] == .45 and source.tolist() == ['fine', 'fine', 'fine', 'coarse']
    assert valid.all()


def test_reliability_gate_zeroes_unsupported_price_after_basis_transform():
    transform = getattr(support_model, 'reliability_gated_centered_basis', None)
    assert transform is not None, 'post-basis reliability gate is missing'
    feature, info = transform(
        values=np.array([.2, .5, .8, .9]),
        valid=np.array([True, True, True, False]),
        reliability=np.array([0., .5, 1., 0.]),
        bandwidth=.25,
    )
    np.testing.assert_array_equal(feature[0], np.zeros(3))
    np.testing.assert_array_equal(feature[3], np.zeros(3))
    np.testing.assert_allclose(feature.sum(axis=0), np.zeros(3), atol=1e-7)
    assert info['zero_reliability_count'] == 1
    assert info['reliability_gate_after_basis'] is True


def test_support_backoff_model_keeps_n_and_v_in_plain_bpr():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder:
        train = pd.DataFrame(dict(
            u_idx=[0, 0, 1, 1], i_idx=[0, 1, 2, 3],
            i_raw=['1', '2', '3', '4'], up=[1., 2., 3., 4.], cat_idx=[0, 0, 0, 0],
        ))
        meta = Path(folder) / 'meta.csv'
        pd.DataFrame(dict(
            PRODUCT_ID=['1', '2', '3', '4'], COMMODITY_DESC=['Food'] * 4,
            SUB_COMMODITY_DESC=['Bread', 'Bread', 'Bread', 'Jam'],
        )).to_csv(meta, index=False)
        features = build_support_backoff_features(
            train, dataset='dunnhumby', meta_path=meta, n_users=2, n_items=5,
            q_n=[.2, .8], q_v=[.8, .2], n_valid=[True, True], v_valid=[True, True],
            shrinkage=10.0,
        )
        assert {k for k in features if k != 'diagnostics'} == {'user_n', 'user_v', PRICE_AXIS}
        assert features['diagnostics'][PRICE_AXIS]['fine_source_items'] == 3
        adj = build_adj(train.u_idx.to_numpy(), train.i_idx.to_numpy(),
                        np.ones(len(train), np.float32), 2, 5)
        torch.manual_seed(43)
        model = SupportBackoffPriceLightGCN(
            n_users=2, n_items=5, adj=adj, features=features, id_dim=8, n_layers=2,
        )
        u, p, n = [torch.tensor(x) for x in ([0, 1], [0, 2], [3, 4])]
        loss, _ = model.bpr_loss(u, p, n)
        loss.backward()
        assert model.encoders['user_n'].weight.grad.norm() > 0
        assert model.encoders['user_v'].weight.grad.norm() > 0
        assert model.encoders[PRICE_AXIS].weight.grad.norm() > 0
        assert set(model.encoders) == {'user_n', 'user_v', PRICE_AXIS}
        try:
            model.bpr_loss(u, p, n, weights=torch.ones(2))
        except ValueError:
            pass
        else:
            raise AssertionError('M2 accepted M4 sample weights')


def test_two_epoch_screen_trains_only_new_m2():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder, patch.object(screen.base.v3, 'DEVICE', 'cpu'):
        prep = prepared()
        d = prep['data']
        train = pd.DataFrame(dict(
            u_idx=d['tr_u'], i_idx=d['tr_i'], i_raw=['1', '2', '2', '3'],
            up=[1., 3., 2., 4.], cat_idx=[0] * 4,
        ))
        meta = Path(folder) / 'meta.csv'
        pd.DataFrame(dict(
            PRODUCT_ID=['1', '2', '3'], COMMODITY_DESC=['Food'] * 3,
            SUB_COMMODITY_DESC=['Bread'] * 3,
        )).to_csv(meta, index=False)
        features = build_support_backoff_features(
            train, dataset='dunnhumby', meta_path=meta, n_users=2, n_items=d['n_items'],
            q_n=prep['q_n'], q_v=prep['q_v'], n_valid=[True, True], v_valid=[True, True],
        )
        features['keys'] = d['pos_key']
        cfg = replace(screen.configure(folder), epochs=2, eval_every=1,
                      patience_start=1, patience=1, batch_size=2)
        prep.update(features=features, features_sha256=screen.feature_hash(features),
                    feature_settings=json.loads(json.dumps(screen.SETTINGS)),
                    screen_config=asdict(cfg), source_report='synthetic',
                    source_report_sha256=screen.source.SOURCE_SHA)
        prep['base_cfg'] = dict(screen.base.v3.CFG, EVAL_BATCH=2, KS=[10, 20, 50])
        prep['cache'] = EvalCache({0: np.array([3, 4]), 1: np.array([4, 5])},
                                  {0: np.array([2., 1.]), 1: np.array([1., 3.])},
                                  np.array([1., 3.]), (1.5, 2.5), d['n_items'])
        prep['meta'] = dict(price_pct=np.linspace(0, 1, d['n_items']),
                            pop_prob=np.ones(d['n_items']) / d['n_items'],
                            cat=np.arange(d['n_items']) % 3)
        m1 = screen.base.capacity._build_model(
            prep, screen.base.capacity.CapacitySearchConfig(),
            dict(model_id=screen.base.capacity.M1_MODEL_ID, id_dim=64, pref_reg=.001), 43,
        )
        metrics = screen.base.capacity._evaluate(m1, prep)
        anchors = [dict(model_id='m1', seed=43, origin='synthetic', metrics=metrics,
                        selected_epoch=1, stopped_epoch=2,
                        curve=[dict(epoch=e, metrics=metrics) for e in (1, 2)])]
        with patch.object(screen, 'configure', return_value=cfg), \
             patch.object(screen, 'read_source', return_value=({}, anchors)):
            paths = screen.run(cfg, prep)
            report = json.loads(Path(paths['json']).read_text())
            assert report['new_fit_count'] == 1
            assert report['new_arms'] == [screen.spec()]
            assert report['arms'][-1]['diagnostics']['nv_activity_ok']
            assert not report['final_test'] and not report['holdout']
            with patch.object(screen.es, '_train', side_effect=AssertionError('must reuse')):
                screen.run(cfg, prep)


if __name__ == '__main__':
    test_support_backoff_price_prefers_fine_and_shrinks_sparse_items()
    test_support_backoff_model_keeps_n_and_v_in_plain_bpr()
    test_two_epoch_screen_trains_only_new_m2()
    print('support-backoff price M2 checks passed')
