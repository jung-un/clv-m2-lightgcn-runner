from dataclasses import replace
import importlib
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('lightgcn_clv_history_linear_nv_early_stop'), 'early-stop runner is not implemented'
    return importlib.import_module('lightgcn_clv_history_linear_nv_early_stop')


def records(values):
    return [dict(epoch=25*(i+1), metrics={'price_purchase_amount_weighted_hit@10': v,
        'recall@10': .01+i*.001}) for i, v in enumerate(values)]


def test_replay_stops_before_late_rebound_and_keeps_earlier_tie():
    m = module()
    r = m.replay(records([1,2,2,2,2,2,2,2,99,99,99,99]), m.configure())
    assert r['stopped_epoch'] == 200
    assert r['selected_epoch'] == 50
    assert r['selected_record']['metrics']['recall@10'] == .011


def test_replay_improvement_resets_patience_and_requires_contiguous_prefix():
    m = module()
    r = m.replay(records([1,1,1,1,1,2,2,2,2,2,9,9]), m.configure())
    assert (r['selected_epoch'], r['stopped_epoch']) == (150,250)
    with pytest.raises(ValueError, match='grid'):
        m.replay(records([1]*12)[1:], m.configure())
    with pytest.raises(ValueError, match='incomplete'):
        m.replay(records([1]*4), m.configure())


def test_replay_rejects_nan():
    m=module()
    with pytest.raises(ValueError, match='finite'):
        m.replay(records([1,np.nan]+[1]*10), m.configure())


def test_default_has_one_seed_two_new_models_no_constant_arms():
    m=module()
    cfg=m.configure()
    assert cfg.seeds == (43,) and cfg.epochs == 300
    assert [s['model_id'] for s in m.specs(cfg)] == ['m2_linear_nv','m5_linear_nv']
    assert len(m.specs(m.configure(include_controls=True))) == 4


def test_baseline_budget_requires_explicit_permission_before_any_fit(monkeypatch):
    m=module()
    cfg=m.configure()
    prepared={'es_config': m.asdict(cfg), 'anchors': [], 'plan': [
        dict(seed=43, model_id='m4', action='train_from_start', max_additional_epochs=300)]}
    with pytest.raises(RuntimeError, match='43:m4'):
        m.run(cfg, prepared=prepared)


def test_seed44_m5_only_schedule_preserves_protocol():
    m=module()
    cfg=m.configure(seeds=(44,), include_m2=False)
    assert [s['model_id'] for s in m.specs(cfg)] == ['m5_linear_nv']
    assert [s['role'] for s in m.anchor_specs(cfg)] == ['M1','M4']
    assert m.preflight(cfg)['new_fits'] == 1
    assert m.preflight(cfg)['selection'] == m.preflight(m.configure())['selection']
    assert m.strength_cfg(cfg).rhos == (0.05,)


def test_curve_reuse_never_uses_later_rebound_and_checks_input_hash(tmp_path):
    import json
    m=module()
    cfg=m.configure(reuse_dirs=(str(tmp_path),))
    old=m.base.capacity.configure_capacity_search(conditions=('baseline',),seeds=(43,))
    key=m.base.capacity._config_hash(old,'input')
    (tmp_path/f'clv_m2_capacity_search_{key}.json').write_text(json.dumps(dict(
        code_version=m.base.capacity.CODE_VERSION,config=m.asdict(old))))
    path=tmp_path/'arms'/key/'baseline_m1_bpr_k1_s43.json'
    path.parent.mkdir(parents=True)
    curve=records([1,2,2,2,2,2,2,2,99,99,99,99])
    for r in curve:
        r['metrics'].update({k:.1 for k in (*m.base.ACCURACY,'vndcg@10')})
    path.write_text(json.dumps(dict(code_version=m.base.capacity.CODE_VERSION,model_id='m1_bpr_k1',
        condition='baseline',seed=43,id_dim=64,pref_reg=.001,rho=0.,axis_dim=0,curve=curve)))
    spec=m.anchor_specs(cfg)[0]
    r=m.reuse_curve({'input_hash':'input'},cfg,spec,43)
    assert r['selected_epoch']==50 and r['stopped_epoch']==200
    assert r['metrics'][m.MONITOR]==2 and len(r['curve'])==8
    assert m.reuse_curve({'input_hash':'changed'},cfg,spec,43) is None


@pytest.mark.parametrize('role',['M2','M5','M1','M4'])
def test_real_training_restores_best_and_resume_does_not_retrain(tmp_path,monkeypatch,role):
    import torch
    from test_lightgcn_clv_history_m5_strength import prepared
    m=module()
    torch.set_num_threads(1)
    monkeypatch.setattr(m.base.v3,'DEVICE','cpu')
    cfg=replace(m.configure(out_dir=str(tmp_path)),epochs=8,eval_every=1,patience_start=4,patience=2,batch_size=2)
    spec=next(s for s in (m.specs(cfg) if role in ('M2','M5') else m.anchor_specs(cfg)) if s['role']==role)
    prep=prepared()
    # Deterministic metric stub makes epoch 2 best; epoch 6 must stop.
    scores=iter([1.,2.,2.,2.,2.,2.])
    snapshots=[]
    def evaluate(model,prep):
        snapshots.append(m.clone_state(model))
        return {**{k:.1 for k in (*m.base.ACCURACY,*m.fixed.PRIMARY)},m.MONITOR:next(scores)}
    monkeypatch.setattr(m.base.capacity,'_evaluate',evaluate)
    build=lambda: (m.fixed._build(prep,m.strength_cfg(cfg),spec,43) if 'linear' in spec['model_id']
                   else m.base._build_model(prep,m.strength_cfg(cfg),spec,43))
    from clv_run_state import ProgressStore,RunIdentity
    store=ProgressStore(tmp_path/'progress',RunIdentity('test',role,43,'test','test','input'))
    original_save=store.save_epoch
    def interrupted(*args,**kw):
        p=original_save(*args,**kw)
        if kw['epoch']==3:
            raise RuntimeError('simulated disconnect after durable checkpoint')
        return p
    store.save_epoch=interrupted
    with pytest.raises(RuntimeError,match='simulated disconnect'):
        m._train(build(),prep,cfg,spec,43,store,tmp_path)
    store.save_epoch=original_save
    model=build()
    result=m._train(model,prep,cfg,spec,43,store,tmp_path)
    assert (result['selected_epoch'],result['stopped_epoch'])==(2,6)
    assert result['training']['resumed_from_epoch']==3
    for name,value in model.state_dict().items():
        expected=snapshots[1][name]
        torch.testing.assert_close(value.to_dense() if value.is_sparse else value,
                                   expected.to_dense() if expected.is_sparse else expected,rtol=0,atol=0)
    assert len(snapshots)==6
    again=m._train(build(),prep,cfg,spec,43,store,tmp_path)
    assert again['stopped_epoch']==6 and len(snapshots)==6


def test_report_distinguishes_incremental_gain_from_business_goal(tmp_path):
    m=module()
    cfg=m.configure(out_dir=str(tmp_path))
    arms=[]
    for spec in m.anchor_specs(cfg)+m.specs(cfg):
        value=1. if spec['role']=='M1' else .8 if spec['role']=='M4' else .9
        metrics={**{k:1. for k in m.base.ACCURACY},**{k:value for k in m.fixed.PRIMARY}}
        arms.append(dict(**spec,seed=43,origin='synthetic',metrics=metrics,
            selected_epoch=50,stopped_epoch=200,stop_reason='patience',curve=[dict(epoch=50,metrics=metrics)]))
    saved=m.save(arms,cfg)
    import pandas as pd
    read=pd.read_csv(saved.attrs['paths']['reading'])
    m5=read[(read.model_id=='m5_linear_nv') & (read.seed=='43')].iloc[0]
    assert m5.directional_pass and not m5.overall_goal_direction
    assert not read.model_id.str.contains('constant').any()
