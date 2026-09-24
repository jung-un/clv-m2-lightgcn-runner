from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch
import clv_linear_nv_original_m4_screen as m
from test_lightgcn_clv_history_m5_strength import prepared


def inputs():
    p = prepared()
    p['user_economic_valid'] = np.ones(2, bool)
    p['item_bin'] = np.arange(p['data']['n_items']) % 4
    p['user_bin_fit'] = np.ones((2,4))
    p['item_amount_percentile'][0] = .5
    return p


def test_original_formula_and_invalid_last_bin_audit():
    p=inputs(); cfg=m.configure('unused')
    frame,w,meta=m.audit_weights(p,cfg)
    u=p['data']['tr_u'];i=p['data']['tr_i']
    raw=1+.5*p['q_c'][u]*p['item_amount_percentile'][i]
    np.testing.assert_allclose(w,raw/raw.mean())
    assert meta['original_invalid_extra_absent']
    p['item_economic_valid'][0]=False;p['item_bin'][0]=-1
    _,_,meta=m.audit_weights(p,cfg)
    assert meta['invalid_item_negative_bin_rows']>0
    assert not meta['original_invalid_extra_absent']


def test_invalid_blocks_before_build(tmp_path,monkeypatch):
    p=inputs();cfg=m.configure(str(tmp_path))
    p.update(screen_config=asdict(cfg),source_report='synthetic')
    p['item_economic_valid'][0]=False;p['item_bin'][0]=-1
    monkeypatch.setattr(m.prior,'verified_anchors',lambda *a:[])
    monkeypatch.setattr(m.base,'_build_model',lambda *a:pytest.fail('must not build'))
    with pytest.raises(RuntimeError,match='무효 입력'):
        m.run(cfg,p)


def test_missing_report_no_data(tmp_path,monkeypatch):
    monkeypatch.setattr(m.base,'_prepare',lambda *a:pytest.fail('must not load'))
    with pytest.raises(ValueError,match='Missing/changed'):
        m.prepare(tmp_path/'missing',tmp_path)


def test_same_linear_model_and_short_training(tmp_path,monkeypatch):
    from clv_run_state import ProgressStore,RunIdentity
    torch.set_num_threads(1);monkeypatch.setattr(m.base.v3,'DEVICE','cpu')
    p=inputs();cfg=m.configure(str(tmp_path))
    old=m.es.fixed._build(p,m.es.strength_cfg(m.prior.configure(str(tmp_path))),m.specs()[1],43)
    new=m.es.fixed._build(p,m.es.strength_cfg(cfg),m.specs()[1],43)
    for a,b in zip(old.parameters(),new.parameters()):
        assert torch.equal(a,b)
    assert new.pref_reg==.001
    cfg=replace(cfg,epochs=2,eval_every=1,patience_start=1,patience=1,batch_size=2)
    metrics={k:1. for k in (*m.base.ACCURACY,*m.es.fixed.PRIMARY)}
    monkeypatch.setattr(m.base.capacity,'_evaluate',lambda *a:metrics)
    _,p['m4_weights'],p['m4_diagnostics']=m.audit_weights(p,cfg)
    arms=[]
    for spec in m.specs():
        root=tmp_path/spec['model_id'];root.mkdir()
        store=ProgressStore(root/'progress',RunIdentity('test',spec['model_id'],43,'a','b','c'))
        build=m.es.fixed._build if spec['model_id']==m.M5 else m.base._build_model
        model=build(p,m.es.strength_cfg(cfg),spec,43)
        result=m.es._train(model,p,cfg,spec,43,store,root)
        assert result['stopped_epoch']==2
        arms.append(dict(**spec,seed=43,origin='synthetic',**result))
    anchors=[dict(arms[0],model_id=mid) for mid in ('m1','m4','m5_linear_nv')]
    p['source_report']='synthetic'
    partial=m.save(anchors+arms[:1],m.configure(str(tmp_path)),p)
    assert not json.loads(open(partial['json']).read())['reading']['complete']
    paths=m.save(anchors+arms,m.configure(str(tmp_path)),p)
    result=json.loads(open(paths['json']).read())
    assert result['reading']['complete']
    assert result['reading']['accuracy_guard_vs_m1']
    assert not result['reading']['both_economic_at10_above_'+m.M4]
