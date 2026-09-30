"""Minimal CPU check for train-only fine-type price, N/V, and plain BPR."""

from tempfile import TemporaryDirectory
from pathlib import Path
from dataclasses import asdict, replace
from unittest.mock import patch
import json

import numpy as np
import pandas as pd
import torch

import clv_m2_fine_price_screen as screen
from clv_m2_fine_price import FINE_AXIS, FinePriceLightGCN, build_fine_features
from lightgcn_clv_v3 import EvalCache, build_adj
from test_lightgcn_clv_history_m5_strength import prepared


def test_fine_price_m2():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder:
        train = pd.DataFrame(dict(u_idx=[0, 0, 1, 1], i_idx=[0, 1, 2, 3],
            i_raw=['1', '2', '3', '4'], up=[1., 2., 3., 4.], cat_idx=[0, 0, 0, 0]))
        hm_meta = Path(folder)/'hm.csv'
        pd.DataFrame(dict(article_id=['0000000001', '0000000002', '0000000003', '0000000004'],
            product_group_name=['Upper']*4, product_type_name=['Shirt']*3+['Vest'])).to_csv(hm_meta, index=False)
        dun_meta = Path(folder)/'dun.csv'
        pd.DataFrame(dict(PRODUCT_ID=['1', '2', '3', '4'], COMMODITY_DESC=['Food']*4,
            SUB_COMMODITY_DESC=['Bread']*3+['Jam'])).to_csv(dun_meta, index=False)
        args = dict(n_users=2, n_items=5, q_n=[.2, .8], q_v=[.8, .2],
            n_valid=[True, True], v_valid=[True, True])
        for dataset, path in [('hm', hm_meta), ('dunnhumby', dun_meta)]:
            features = build_fine_features(train, dataset=dataset, meta_path=path, **args)
            assert features['diagnostics'][FINE_AXIS]['valid_item_count'] == 3
            assert np.any(features[FINE_AXIS][:3])
            assert not np.any(features[FINE_AXIS][3:])  # Singleton + no train observation.
            adj = build_adj(train.u_idx.to_numpy(), train.i_idx.to_numpy(),
                np.ones(len(train), np.float32), 2, 5)
            torch.manual_seed(43)
            model = FinePriceLightGCN(n_users=2, n_items=5, adj=adj,
                features=features, id_dim=8, n_layers=2)
            u, p, n = [torch.tensor(x) for x in ([0, 1], [0, 2], [3, 4])]
            loss, _ = model.bpr_loss(u, p, n)
            loss.backward()
            assert model.encoders['user_n'].weight.grad.norm() > 0
            assert model.encoders['user_v'].weight.grad.norm() > 0
            assert model.fine_encoder.weight.grad.norm() > 0
            assert model.representation_diagnostics()['shared_parameters'] == 5*3*8
            regularization = model.batch_l2(u, p, n)
            torch.testing.assert_close(model.batch_l2(u.repeat(2), p.repeat(2), n.repeat(2)), regularization)
            try:
                model.bpr_loss(u, p, n, weights=torch.ones_like(u, dtype=torch.float))
            except ValueError:
                pass
            else:
                raise AssertionError('M2 accepted M4 sample weights')


def test_two_epoch_screen():
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder, patch.object(screen.base.v3, 'DEVICE', 'cpu'):
        prep = prepared()
        d = prep['data']
        train = pd.DataFrame(dict(u_idx=d['tr_u'], i_idx=d['tr_i'],
            i_raw=['1', '2', '2', '3'], up=[1., 3., 2., 4.], cat_idx=[0]*4))
        meta_path = Path(folder)/'product.csv'
        pd.DataFrame(dict(PRODUCT_ID=['1', '2', '3'], COMMODITY_DESC=['Food']*3,
            SUB_COMMODITY_DESC=['Bread']*3)).to_csv(meta_path, index=False)
        features = build_fine_features(train, dataset='dunnhumby', meta_path=meta_path,
            n_users=2, n_items=d['n_items'], q_n=prep['q_n'], q_v=prep['q_v'],
            n_valid=[True, True], v_valid=[True, True])
        features['keys'] = d['pos_key']
        cfg = replace(screen.configure(folder), epochs=2, eval_every=1,
            patience_start=1, patience=1, batch_size=2)
        prep.update(features=features, features_sha256=screen.feature_hash(features),
            feature_settings=json.loads(json.dumps(screen.SETTINGS)),
            screen_config=asdict(cfg), source_report='synthetic',
            source_report_sha256=screen.source.SOURCE_SHA)
        prep['base_cfg'] = dict(screen.base.v3.CFG, EVAL_BATCH=2, KS=[10, 20, 50])
        gt = {0: np.array([3, 4]), 1: np.array([4, 5])}
        amounts = {0: np.array([2., 1.]), 1: np.array([1., 3.])}
        prep['cache'] = EvalCache(gt, amounts, np.array([1., 3.]), (1.5, 2.5), d['n_items'])
        prep['meta'] = dict(price_pct=np.linspace(0, 1, d['n_items']),
            pop_prob=np.ones(d['n_items'])/d['n_items'], cat=np.arange(d['n_items']) % 3)
        m1 = screen.base.capacity._build_model(prep, screen.base.capacity.CapacitySearchConfig(),
            dict(model_id=screen.base.capacity.M1_MODEL_ID, id_dim=64, pref_reg=.001), 43)
        metrics = screen.base.capacity._evaluate(m1, prep)
        anchors = [dict(model_id='m1', seed=43, origin='synthetic', metrics=metrics,
            selected_epoch=1, stopped_epoch=2,
            curve=[dict(epoch=e, metrics=metrics) for e in (1, 2)])]
        prep['previous_arm'] = dict(anchors[0], model_id=screen.prior.MODEL_ID,
            identity={'input_hash': 'synthetic'})
        prep['previous_report_sha256'] = 'synthetic'
        with patch.object(screen, 'configure', return_value=cfg), \
             patch.object(screen, 'read_source', return_value=({}, anchors)):
            paths = screen.run(cfg, prep)
            report = json.loads(Path(paths['json']).read_text())
            assert report['new_fit_count'] == 1 and report['code_version'] == screen.VERSION
            assert report['arms'][-1]['diagnostics']['item_fine_within_active']
            assert len(pd.read_csv(paths['comparison'])) == 2*len(metrics)
            assert len(pd.read_csv(paths['same_epoch_comparison'])) == 4*len(metrics)
            assert not report['final_test'] and not report['holdout']
            with patch.object(screen.es, '_train', side_effect=AssertionError('must reuse')):
                screen.run(cfg, prep)


if __name__ == '__main__':
    test_fine_price_m2()
    test_two_epoch_screen()
    print('fine-type price M2 CPU check passed')
