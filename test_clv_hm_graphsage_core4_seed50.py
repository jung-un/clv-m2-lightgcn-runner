from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch

import clv_hm_graphsage_core4_seed50 as s
import clv_hm_graphsage_final8_seed49 as old
import clv_personal_graphsage_checkpoint as disk
from test_clv_hm_graphsage_final8_seed49 import prepared as gs_fixture, reference_fixture
from test_clv_ngcf_school_storage import api, Request

def test_hm_prepare_routes_inputs_and_graph_without_training(prepared, monkeypatch):
    p = prepared
    p.update(q_n=np.full(3,.5,np.float32), q_v=np.full(3,.5,np.float32),
             item_amount_percentile=np.full(5,.5,np.float32), observed_weight_audit={'sha256':'weight'})
    p['data'].update(splits={'test':{}}, data_stats={'same':True}, train={})
    p['base_cfg'] = {}
    gc = s.base.hm.configure()
    common = {'input_hash':p['input_hash'], 'data_stats':p['data']['data_stats'],
              'feature_sha256':p['feature_sha256'], 'split':'hm-lastweek',
              'm4_c_weight_audit':{'sha256':'weight'}, 'm3_audit':{'cv':.2}, 'fixed':{},
              'config':{k:getattr(gc,k) for k in s.hm_schema.GRAPH_KEYS}}
    refs = {**p['references'], 'sources':{k:{'protocol':common} for k in ('m2','m4')},'audit':{}}
    monkeypatch.setattr(s.base,'load_references',lambda cfg:refs)
    monkeypatch.setattr(s,'verify_input_files',lambda expected: {})
    monkeypatch.setattr(s.base,'_prepare_hm_inputs',lambda cfg:dict(p))
    monkeypatch.setattr(s.base.hm,'validate_split',lambda data:None)
    monkeypatch.setattr(s.base.hm.hm_graph,'_purchase_keyed',lambda data:None)
    monkeypatch.setattr(s.base.fixed.graph,'centered_edge_signals',lambda *a:p['signals'])
    monkeypatch.setattr(s.base.fixed,'build_graph',lambda *a:{**p['observed_graph'],'audit':{'cv':.2}})
    monkeypatch.setattr(s.execution,'train',lambda *a,**kw:pytest.fail('prepare trained'))
    monkeypatch.setattr(s.base.fixed,'_evaluate',lambda *a:pytest.fail('prepare evaluated'))
    actual=s.prepare(p['cfg'])
    assert actual['cfg'].dataset=='hm' and actual['base_cfg']['SEED_LIST']==[50]
    assert actual['protocol']['new_fit_count']==4
    assert actual['observed_weights'] is p['observed_weights']
    assert actual['features'] is p['features']
    torch.testing.assert_close(actual['graphsage_m3_adj'].to_dense(),p['graphsage_m3_adj'].to_dense())
    assert s.report(actual)['absolute'].empty


@pytest.fixture
def prepared(gs_fixture):
    p = gs_fixture
    p['cfg'] = s.configure(out_dir=str(p['run_dir']))
    return p


def test_only_core_seed50_and_unchanged_scientific_settings(prepared):
    cfg = prepared['cfg']
    assert s.ORDER == ('M1', 'M2', 'M3', 'M4') and cfg.seeds == (50,)
    prior = asdict(old.configure())
    current = asdict(cfg)
    for k in current.keys() - {'out_dir', 'seeds'}:
        assert current[k] == prior[k]
    for change in ({'seeds': (49,)}, {'epochs': 299}, {'batch_size': 8192}, {'lr': .01}):
        with pytest.raises(ValueError): s.validate_config(replace(cfg, **change))
    with pytest.raises(ValueError): s.identity(prepared, 'M5_A')
    assert s.identity(prepared, 'M1').seed == 50
    assert s.arm_path(prepared, 'M1').name.endswith('_s50.json')


@pytest.mark.parametrize('label', s.ORDER)
def test_unchanged_forward_and_initialization(prepared, label):
    torch.manual_seed(50)
    a, wa = old.build_model(prepared, label)
    torch.manual_seed(50)
    b, wb = s.build_model(prepared, label)
    assert wa is wb
    for k,v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
    u,p,n = torch.tensor([0,1,2]),torch.tensor([1,2,3]),torch.tensor([4,4,0])
    for x,y in zip(a._pair_scores(u,p,n), b._pair_scores(u,p,n)):
        torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_four_fits_and_evaluations_then_reuse(prepared, monkeypatch):
    monkeypatch.setattr(s.SafeProgressStore, '_personal_storage_installed', True, raising=False)
    calls=[]
    def train(model,p,cfg,model_id,seed,store,**kwargs):
        assert seed == 50 and cfg.epochs == 300
        calls.append(model_id)
        return [{'epoch': i} for i in range(1,301)]
    monkeypatch.setattr(s.execution,'train',train)
    evaluations=[]
    def evaluate(*a):
        evaluations.append(1)
        return dict.fromkeys(prepared['references']['metrics'], 1.01)
    monkeypatch.setattr(s.base.fixed,'_evaluate',evaluate)
    assert s.report(prepared)['absolute'].empty
    result=s.run(prepared)
    assert result['reading']['complete'] and result['reading']['planned'] == 4
    assert not result['reading']['full_eight_model_panel_complete']
    assert set(result['absolute'].seed) == {50}
    s.run(prepared)
    assert len(calls)==len(evaluations)==4


def test_requires_storage_install(prepared):
    with pytest.raises(RuntimeError, match='저장 셀'):
        s.run_arm(prepared,'M1')


def personal_api(api, monkeypatch):
    original_get=api.get
    def get(fileId=None,fields=None):
        if fileId is not None: return original_get(fileId=fileId,fields=fields)
        return Request(lambda: {'user':{'emailAddress':disk.ACCOUNT},
            'storageQuota':{'limit':str(100*10**9),'usage':'0'}})
    monkeypatch.setattr(api,'get',get)
    import googleapiclient.http as http
    original_download=http.MediaIoBaseDownload
    monkeypatch.setattr(http,'MediaIoBaseDownload',lambda target,data,**kw:original_download(target,data))
    return api


def test_storage_rotation_resume_exact_and_isolated(prepared, api, monkeypatch, tmp_path):
    personal_api(api,monkeypatch)
    old_cls=old.SafeProgressStore
    monkeypatch.setattr(s,'SafeProgressStore',s.SafeProgressStore)
    parent=disk.install(s,api,prepared,local_root=tmp_path/'local')
    assert old.SafeProgressStore is old_cls
    cfg=replace(prepared['cfg'],epochs=3,batch_size=4)
    ident=s.identity(prepared,'M2')
    def fresh():
        s.base.v3.set_seed(50)
        model,weights=s.build_model(prepared,'M2')
        s.base.v3.set_seed(50)
        return model,weights
    full,w=fresh()
    baseline=old.SafeProgressStore(tmp_path/'baseline',ident)
    hist=s.execution.train(full,prepared,cfg,ident.model_id,50,baseline,
                          row_weights=w,negative_sampler=s.sample_uniform_negative_matrix)
    partial,w=fresh()
    store=s.SafeProgressStore(prepared['run_dir']/'progress',ident)
    s.execution.train(partial,prepared,replace(cfg,epochs=2),ident.model_id,50,store,
                      row_weights=w,negative_sampler=s.sample_uniform_negative_matrix)
    store.latest_checkpoint.unlink()  # Simulate lost runtime-local disk.
    resumed,w=fresh()
    actual=s.execution.train(resumed,prepared,cfg,ident.model_id,50,store,
                            row_weights=w,negative_sampler=s.sample_uniform_negative_matrix)
    assert actual==hist
    for k,v in full.state_dict().items():
        torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)
    rows=disk.backend().rows_for(api,parent,ident)
    assert [int(r['appProperties']['clv_epoch']) for r in rows]==[3,2]
    assert len(api.deleted)==1
    full_p=torch.load(baseline.latest_checkpoint,weights_only=False)
    saved=torch.load(store.latest_checkpoint,weights_only=False)
    assert full_p['numpy_rng_state']==saved['numpy_rng_state']
    assert torch.equal(full_p['torch_rng_state'],saved['torch_rng_state'])
    for k,values in full_p['optimizer_state']['state'].items():
        for name,v in values.items():
            torch.testing.assert_close(v,saved['optimizer_state']['state'][k][name],atol=0,rtol=0)


def test_personal_quota_no_drive_enumeration_and_wrong_account(api, monkeypatch):
    personal_api(api,monkeypatch)
    monkeypatch.setattr(api,'list',lambda **kw:pytest.fail('whole Drive scan'))
    assert disk.backend().check_space(api,initial=True)==100*10**9
    monkeypatch.setattr(api,'get',lambda **kw:Request(lambda:{'user':{'emailAddress':'school'}}))
    with pytest.raises(RuntimeError,match='개인 Drive'): disk.backend().check_space(api)
