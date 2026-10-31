from dataclasses import asdict
import pytest
import torch
import clv_hm_graphsage_final8_seed49 as old
import clv_hm_graphsage_final8_seed51 as new
import clv_hm_gat_final8_seed49 as gat49
import clv_hm_gat_final8_seed51 as gat51
from test_clv_hm_graphsage_final8_seed49 import prepared, reference_fixture

def test_conditions():
    for a,b in ((old,new),(gat49,gat51)):
        x,y=asdict(a.configure()),asdict(b.configure())
        assert y['seeds']==(51,) and b.ORDER==a.ORDER and len(b.ORDER)==8
        assert {k:v for k,v in x.items() if k not in ('seeds','out_dir')} == {k:v for k,v in y.items() if k not in ('seeds','out_dir')}

@pytest.mark.parametrize('label',new.ORDER)
def test_csr_equivalence(prepared,label):
    prepared['cfg']=new.configure(out_dir=str(prepared['run_dir']))
    torch.manual_seed(51); a,w=old.build_model(prepared,label)
    torch.manual_seed(51); b,wb=new.build_model(prepared,label)
    assert w is wb and b.adj.layout==torch.sparse_csr
    torch.testing.assert_close(a.adj.to_dense(),b.adj.to_dense(),atol=0,rtol=0)
    u,pos,neg=torch.tensor([0,1,2]),torch.tensor([1,2,3]),torch.tensor([[4],[4],[0]])
    weights=torch.tensor(w[:3],dtype=torch.float32) if w is not None else None
    optimizers=[torch.optim.Adam(m.parameters(),lr=.0005) for m in (a,b)]
    losses=[]
    for model,opt in zip((a,b),optimizers):
        loss,_,_=old.base.fixed.recheck._batch_loss(model,u,pos,neg,weights)
        loss.backward(); losses.append(loss)
    torch.testing.assert_close(losses[0],losses[1],atol=2e-6,rtol=2e-4)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
    for opt in optimizers: opt.step()
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-4)
