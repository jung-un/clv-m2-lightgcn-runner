from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch

import clv_hm_gat_m5_seed50 as s
import clv_hm_gat_final8_seed49 as old
from test_clv_hm_gat_final8_seed49 import prepared as gs_fixture, reference_fixture

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
    monkeypatch.setattr(s,'compare_m3_audit',lambda *a: {})
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
    torch.testing.assert_close(actual['gat_m3_prior'].to_dense(),p['gat_m3_prior'].to_dense())
    assert s.report(actual)['absolute'].empty


@pytest.fixture
def prepared(gs_fixture):
    p = gs_fixture
    p['cfg'] = s.configure(out_dir=str(p['run_dir']))
    return p


def test_only_core_seed50_and_unchanged_scientific_settings(prepared):
    cfg = prepared['cfg']
    assert s.ORDER == ('M5_A', 'M5_B', 'M5_C', 'M5_D') and cfg.seeds == (50,)
    prior = asdict(old.configure())
    current = asdict(cfg)
    for k in current.keys() - {'out_dir', 'seeds'}:
        assert current[k] == prior[k]
    for change in ({'seeds': (49,)}, {'epochs': 299}, {'batch_size': 8192}, {'lr': .01}):
        with pytest.raises(ValueError): s.validate_config(replace(cfg, **change))
    with pytest.raises(ValueError): s.identity(prepared, 'M1')
    assert s.identity(prepared, 'M5_A').seed == 50
    assert s.arm_path(prepared, 'M5_A').name.endswith('_s50.json')


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
    monkeypatch.setattr(s.execution,'compact_completed_checkpoint',lambda *a:None)
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


def test_checkpoint_restores_model_adam_and_rng(prepared, tmp_path):
    store = s.SafeProgressStore(tmp_path/'progress', s.identity(prepared, 'M5_A'))
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    rng = np.random.default_rng(50)
    store.save_epoch(model, optimizer, rng, epoch=1, max_epoch=300, history=[{'epoch': 1}])
    expected = {k:v.clone() for k,v in model.state_dict().items()}
    expected_random = rng.random(4)
    expected_torch = torch.rand(4)
    for p in model.parameters():
        p.data.zero_()
    restored = store.restore_epoch(model, optimizer, rng)
    assert restored['next_epoch'] == 2
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k], atol=0, rtol=0)
    np.testing.assert_array_equal(rng.random(4), expected_random)
    torch.testing.assert_close(torch.rand(4), expected_torch, atol=0, rtol=0)
    assert all(int(state['step']) == 1 for state in optimizer.state.values())


def test_corrupt_checkpoint_does_not_restart(prepared, tmp_path):
    store = s.SafeProgressStore(tmp_path/'progress', s.identity(prepared, 'M5_A'))
    store.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    store.latest_checkpoint.touch()
    model = torch.nn.Linear(2, 1)
    with pytest.raises(RuntimeError, match='자동 처음부터'):
        store.restore_epoch(model, torch.optim.Adam(model.parameters()), np.random.default_rng(50))


