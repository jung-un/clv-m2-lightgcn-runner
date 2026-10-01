"""One integration check for the post-basis reliability-gated M2 screen."""

from dataclasses import asdict, replace
from importlib import import_module, util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

import numpy as np
import pandas as pd
import torch

from clv_m2_support_backoff_price import build_support_backoff_features
from lightgcn_clv_v3 import EvalCache
from test_lightgcn_clv_history_m5_strength import prepared


def test_two_epoch_gated_screen_trains_one_internal_m2():
    assert util.find_spec('clv_m2_reliability_gated_price_screen') is not None
    screen = import_module('clv_m2_reliability_gated_price_screen')
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
            train, dataset='dunnhumby', meta_path=meta, n_users=2,
            n_items=d['n_items'], q_n=prep['q_n'], q_v=prep['q_v'],
            n_valid=[True, True], v_valid=[True, True],
            reliability_after_basis=True,
        )
        assert features['diagnostics']['item_support_backoff_price'][
            'reliability_gate_after_basis'] is True
        features['keys'] = d['pos_key']
        cfg = replace(screen.configure(folder), epochs=2, eval_every=1,
                      patience_start=1, patience=1, batch_size=2)
        prep.update(
            features=features,
            features_sha256=screen.feature_hash(features),
            feature_settings=json.loads(json.dumps(screen.SETTINGS)),
            screen_config=asdict(cfg), source_report='synthetic',
            source_report_sha256=screen.source.SOURCE_SHA,
        )
        prep['base_cfg'] = dict(screen.base.v3.CFG, EVAL_BATCH=2, KS=[10, 20, 50])
        prep['cache'] = EvalCache(
            {0: np.array([3, 4]), 1: np.array([4, 5])},
            {0: np.array([2., 1.]), 1: np.array([1., 3.])},
            np.array([1., 3.]), (1.5, 2.5), d['n_items'],
        )
        prep['meta'] = dict(
            price_pct=np.linspace(0, 1, d['n_items']),
            pop_prob=np.ones(d['n_items']) / d['n_items'],
            cat=np.arange(d['n_items']) % 3,
        )
        m1 = screen.base.capacity._build_model(
            prep, screen.base.capacity.CapacitySearchConfig(),
            dict(model_id=screen.base.capacity.M1_MODEL_ID, id_dim=64, pref_reg=.001), 43,
        )
        metrics = screen.base.capacity._evaluate(m1, prep)
        anchors = [dict(
            model_id='m1', seed=43, origin='synthetic', metrics=metrics,
            selected_epoch=1, stopped_epoch=2,
            curve=[dict(epoch=e, metrics=metrics) for e in (1, 2)],
        )]
        with patch.object(screen, 'configure', return_value=cfg), \
             patch.object(screen, 'read_source', return_value=({}, anchors)):
            paths = screen.run(cfg, prep)
        report = json.loads(Path(paths['json']).read_text())
        assert report['code_version'] == screen.VERSION
        assert report['new_arms'] == [screen.spec()]
        assert report['new_fit_count'] == 1
        assert report['feature_diagnostics']['item_support_backoff_price'][
            'reliability_gate_after_basis'] is True
        assert not report['final_test'] and not report['holdout']

