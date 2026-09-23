import json
from dataclasses import replace
import numpy as np
import pytest
import torch

import clv_linear_nv_affine_l2_screen as m
from test_lightgcn_clv_history_m5_strength import prepared


def test_only_affine_l2_changes_forward_initialization_and_other_gradients_identical(monkeypatch):
    monkeypatch.setattr(m.base.v3,'DEVICE','cpu')
    p=prepared();cfg=m.configure('unused')
    old=m.es.fixed._build(p,m.es.strength_cfg(cfg),m.es.specs(cfg)[0],43)
    new=m.build(p,cfg)
    for (n,a),(n2,b) in zip(old.named_parameters(),new.named_parameters()):
        assert n==n2 and torch.equal(a,b)
    u,pos,neg=torch.tensor([0,1]),torch.tensor([0,2]),torch.tensor([3,4])
    for a,b in zip(old._pair_scores(u,pos,neg),new._pair_scores(u,pos,neg)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    old.batch_l2(u,pos,neg).backward();new.batch_l2(u,pos,neg).backward()
    for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(b.grad,a.grad*(.1 if '_transform.' in name else 1))


def test_weight_mass_partitions_and_formula(monkeypatch):
    p=prepared();p['axes']={'clv_proxy':np.array([1.,10.])};p['base_cfg']={'SEG_EDGES':(.2,.8)}
    cfg=m.configure('unused')
    p['m4_weights'],_=m.base.weights_module.row_weights(p,cfg,'complementary')
    frame,meta=m.weight_audit(p,cfg)
    for _,part in frame.groupby('grouping'):
        assert part.n_rows.sum()==4
        assert part.row_share.sum()==pytest.approx(1)
        assert part.weight_share.sum()==pytest.approx(1)
        assert part.extra_share.sum()==pytest.approx(1)
    assert meta['no_training']
    p['m4_weights']=np.ones(4)
    with pytest.raises(ValueError,match='match actual'):
        m.weight_audit(p,cfg)


def test_missing_report_stops_before_data_or_training(tmp_path,monkeypatch):
    monkeypatch.setattr(m.base,'_prepare',lambda _:pytest.fail('must fail before data preparation'))
    with pytest.raises(ValueError,match='Missing/changed'):
        m.prepare(tmp_path/'missing.json',tmp_path/'out')


def test_train_resume_and_save_one_candidate(tmp_path,monkeypatch):
    from clv_run_state import ProgressStore,RunIdentity
    torch.set_num_threads(1);monkeypatch.setattr(m.base.v3,'DEVICE','cpu')
    p=prepared();cfg=replace(m.configure(tmp_path),epochs=8,eval_every=1,patience_start=4,patience=2,batch_size=2)
    scores=iter([1,2,2,2,2,2]); calls=[]
    def evaluate(model,prep):
        calls.append(1)
        return {**{k:1. for k in (*m.base.ACCURACY,*m.es.fixed.PRIMARY)},m.es.MONITOR:float(next(scores))}
    monkeypatch.setattr(m.base.capacity,'_evaluate',evaluate)
    store=ProgressStore(tmp_path/'progress',RunIdentity('test',m.MODEL_ID,43,'x','x','synthetic'))
    original=store.save_epoch
    def interrupt(*args,**kw):
        original(*args,**kw)
        if kw['epoch']==3:raise RuntimeError('interrupt')
    store.save_epoch=interrupt
    with pytest.raises(RuntimeError,match='interrupt'):
        m.es._train(m.build(p,cfg),p,cfg,m._spec(),43,store,tmp_path)
    store.save_epoch=original
    result=m.es._train(m.build(p,cfg),p,cfg,m._spec(),43,store,tmp_path)
    assert (result['selected_epoch'],result['stopped_epoch'])==(2,6)
    assert result['training']['resumed_from_epoch']==3 and len(calls)==6
    arms=[]
    for mid,role in [('m1','M1'),('m4','M4'),('m5_linear_nv','M5'),(m.MODEL_ID,'M5')]:
        arms.append(dict(model_id=mid,role=role,seed=43,rho=.05,origin='synthetic',**result))
    p['source_report']='synthetic'
    paths=m.save(arms,m.configure(tmp_path),p)
    report=json.loads(open(paths['json']).read())
    assert report['reading']['accuracy_guard']
    assert not report['reading']['both_economic_at10_above_m5_linear_nv']
    import pandas as pd
    comparison=pd.read_csv(paths['comparison'])
    assert 'm5_linear_nv' in set(comparison[comparison.model_id==m.MODEL_ID].reference)


def test_verified_report_selection_source_and_input_checks(tmp_path,monkeypatch):
    cfg=m.configure(tmp_path);p=prepared()
    metrics={k:1. for k in (*m.base.ACCURACY,*m.es.fixed.PRIMARY)}
    curve=[dict(epoch=i,metrics=metrics) for i in range(25,201,25)]
    arms=[]
    for spec in m.es.anchor_specs(cfg)+m.es.specs(cfg):
        arms.append(dict(**spec,seed=43,selected_epoch=25,stopped_epoch=200,metrics=metrics,curve=curve,
            identity=m.es._identity(p,cfg,spec,43)))
    report=tmp_path/'source.json'
    report.write_text(json.dumps(dict(preflight=m.es.preflight(cfg),arms=arms)))
    monkeypatch.setattr(m,'EXPECTED_REPORT_SHA',m.file_sha256(report))
    monkeypatch.setattr(m.es,'reuse_curve',lambda *a:dict(metrics=metrics))
    assert len(m.verified_anchors(report,cfg,p))==3
    with pytest.raises(ValueError,match='Input hash'):
        m.verified_anchors(report,cfg,dict(p,input_hash='changed'))
    report.write_text('{}')
    with pytest.raises(ValueError,match='missing or changed'):
        m.verified_anchors(report,cfg,p)
