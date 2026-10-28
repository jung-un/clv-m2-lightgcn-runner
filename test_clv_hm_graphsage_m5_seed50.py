from dataclasses import asdict, replace
import pytest
import torch
import clv_hm_graphsage_m5_seed50 as s
import clv_hm_graphsage_final8_seed49 as old
from test_clv_hm_graphsage_final8_seed49 import prepared, reference_fixture

def test_fixed_partition():
    assert s.ORDER == ('M5_A','M5_B','M5_C','M5_D')
    a,b=asdict(s.configure()),asdict(old.configure())
    assert a['seeds']==(50,)
    assert {k:v for k,v in a.items() if k not in ('seeds','out_dir')} == {k:v for k,v in b.items() if k not in ('seeds','out_dir')}
    for change in ({'seeds':(49,)},{'epochs':299},{'batch_size':8192}):
        with pytest.raises(ValueError): s.validate_config(replace(s.configure(),**change))

@pytest.mark.parametrize('label',s.ORDER)
def test_exact_original_model(prepared,label):
    prepared['cfg']=s.configure(out_dir=str(prepared['run_dir']))
    torch.manual_seed(50); a,wa=old.build_model(prepared,label)
    torch.manual_seed(50); b,wb=s.build_model(prepared,label)
    assert wa is wb
    for key,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
    u,p,n=torch.tensor([0,1,2]),torch.tensor([1,2,3]),torch.tensor([4,4,0])
    for x,y in zip(a._pair_scores(u,p,n),b._pair_scores(u,p,n)):
        torch.testing.assert_close(x,y,atol=0,rtol=0)

def test_only_four_fits_then_reuse(prepared,monkeypatch):
    prepared['cfg']=s.configure(out_dir=str(prepared['run_dir']))
    monkeypatch.setattr(s.SafeProgressStore,'_runpod_storage_installed',True,raising=False)
    calls=[]
    def train(model,p,cfg,mid,seed,store,**kwargs):
        assert seed==50 and cfg.epochs==300
        calls.append(mid)
        return [{'epoch':i} for i in range(1,301)]
    monkeypatch.setattr(s.execution,'train',train)
    monkeypatch.setattr(s.base.fixed,'_evaluate',lambda *a:dict.fromkeys(prepared['references']['metrics'],1.0))
    assert s.run(prepared)['complete']
    assert s.run(prepared)['complete']
    assert len(calls)==4
    assert s.report(prepared)['reading']['partition']=='M5_A-M5_D'
