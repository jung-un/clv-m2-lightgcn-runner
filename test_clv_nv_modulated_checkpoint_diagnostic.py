"""Bounded CPU checks of real model scoring, all-metric replay and no-training audit."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import clv_nv_modulated_checkpoint_diagnostic as audit
from clv_shared_nv_feature_model import NVModulatedLightGCN, build_features


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    frame = pd.DataFrame(dict(u_idx=[0, 0, 1, 1, 2, 2], i_idx=[0, 1, 1, 2, 3, 4],
                              up=[1., 2., 3., 4., 5., 6.]))
    f = build_features(frame, n_users=3, n_items=64, q_n=np.array([.1, .5, .9]),
                       q_v=np.array([.2, .6, .8]), valid=np.ones(3, bool))
    # Same sparse propagation/feature implementation as the selected model.
    u, i = frame.u_idx.to_numpy(copy=True), frame.i_idx.to_numpy(copy=True)
    row, col = np.r_[u, i+3], np.r_[i+3, u]
    adj = torch.sparse_coo_tensor(np.stack([row, col]), np.full(len(row), .2, np.float32),
                                   (67, 67), check_invariants=True)
    m = NVModulatedLightGCN(n_users=3, n_items=64, features=f, adj=adj, shared_l2=.0001)
    with torch.no_grad():
        for layer in m.encoders.values():
            layer.weight.copy_(torch.randn_like(layer.weight)*.7)
    cache = audit.source.base.v3.EvalCache(
        {k:np.array(v) for k,v in {0:[10,20], 1:[30], 2:[40,50]}.items()},
        {k:np.array(v) for k,v in {0:[2.,3.], 1:[4.], 2:[5.,6.]}.items()},
        np.array([.1, .5, .9]), (.3, .7), 64)
    d = dict(n_users=3, n_items=64, adj=adj, tr_u=u, tr_i=i, pos_key=f['keys'],
              csr_ptr=np.array([0, 2, 4, 6]), csr_items=i)
    prep = dict(data=d, features=f, features_sha256=audit.source.previous.feature_hash(f),
        feature_settings=dict(audit.source.SETTINGS), input_hash='synthetic', cache=cache,
        base_cfg={'EVAL_BATCH':2, 'K_LIST':[10,20,50]},
        meta=dict(price_pct=np.linspace(0,1,64), pop_prob=np.full(64,1/64), cat=np.arange(64)%4))
    return m, prep


def test_real_model_views_readback_and_full_audit_without_training(tmp_path):
    model, prep = fixture()
    with patch.object(audit.source.base.v3, 'DEVICE', 'cpu'):
        expected = audit.source.base.capacity._evaluate(model, prep)
        assert len(expected) == 142
        before = {k: v.clone() for k, v in model.state_dict().items()}
        views = audit.view_embeddings(model)
        tops = audit.rank_views(views, prep)
        for top in tops.values():
            for u, row in zip(prep['cache'].users, top):
                assert not np.isin(row, prep['data']['csr_items'][2*u:2*u+2]).any()
        absolute = audit.shared._metric_rows(prep, tops)
        assert len(audit.metric_readback(absolute, expected)) == 142
        broken = dict(expected, **{'recall@10':expected['recall@10']+1})
        with pytest.raises(ValueError, match='not reproduced'):
            audit.metric_readback(absolute, broken)
        probe, summary = audit.input_shift(model, prep, views)
        assert len(probe) == 6
        assert summary.set_index('buyer_group').loc['1', 'n_full_valid_loo_missing_share'] == 1
        assert not probe[['n_loo_valid', 'v_loo_valid']].all().all()
        checkpoint = tmp_path/'selected_epoch100.pt'
        torch.save(dict(epoch=100, model_state=before), checkpoint)
        arm = dict(model_id=audit.source.MODEL_ID, seed=43, metrics=expected, selected_epoch=100,
                    checkpoint_sha256=audit.file_sha256(checkpoint))
        context = dict(checkpoint=checkpoint, arm=arm, prepared=prep, cfg=audit.source.configure(str(tmp_path)),
            report={'arms':[dict(model_id='m1',seed=43,metrics=expected)]},
            report_path='synthetic CPU check, not research performance', out_dir=tmp_path/'audit')
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('No optimizer allowed')), \
             patch.object(audit.source.es, '_train', side_effect=AssertionError('No training allowed')):
            paths = audit.run(context)
        result = json.loads(Path(paths['diagnostic']).read_text())
        assert result['optimizer_updates'] == 0 and not result['new_training']
        assert not result['final_test'] and not result['holdout']
        assert result['original_metric_count'] == 142
        assert len(pd.read_csv(paths['comparison'])) == 5*208
        assert len(pd.read_csv(paths['movement_summary'])) == 5*3*4
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        torch.save(dict(epoch=200,model_state=before), checkpoint)
        with pytest.raises(ValueError, match='Checkpoint changed'):
            audit.load_model(context)
    bad = tmp_path/'wrong.json'
    bad.write_text('{}')
    with patch.object(audit.source.base, '_prepare', side_effect=AssertionError('No data load')):
        with pytest.raises(ValueError, match='완료 result'):
            audit.prepare(bad, tmp_path/'unused')


def test_cutoff20_truth_gain_and_loss_reconcile():
    _, prep = fixture()
    tops = {name: np.tile(np.arange(5,55), (3,1)) for name in audit.VIEWS}
    # User0 truth10 exits @10 and truth20 exits @20; move both to beyond50.
    tops['full'][0,5] = 55
    tops['full'][0,15] = 56
    truth, users, summary = audit.movements(tops, prep)
    row = summary[(summary.view == 'full') & (summary.reference == 'id_only')
                  & (summary.segment == '전체') & (summary.cutoff == 20)].iloc[0]
    assert row.truth_lost == 2 and row.weighted_net == pytest.approx(-5/3)
    assert row.recall_net == pytest.approx(-1/3)
    assert 'item_training_buyers' in truth
    assert users.truth_gained.sum() >= 0
