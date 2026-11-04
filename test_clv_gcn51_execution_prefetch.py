import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import clv_hm_graphsage_training_io as execution
from clv_gcn51_execution_prefetch import Batches, optimized_train

class Model(torch.nn.Module):
    m2_active = False
    def __init__(self):
        super().__init__(); self.e = torch.nn.Embedding(23, 4)
    def _pair_scores(self, u, p, n):
        return (self.e(u)*self.e(p)).sum(1), (self.e(u)*self.e(n)).sum(1)
    def batch_l2(self, u,p,n):
        return .001*sum(self.e(x).square().sum() for x in (u,p,n))/len(u)

class Store:
    def __init__(self, root, saved=None):
        self.root=Path(root); self.saved=copy.deepcopy(saved)
        (self.root/'timing').mkdir(exist_ok=True)
    def restore_epoch(self, m,o,r):
        if self.saved is None:return None
        s=self.saved;m.load_state_dict(s['model']);o.load_state_dict(s['opt']);r.bit_generator.state=s['rng']
        torch.set_rng_state(s['torch_rng'])
        return dict(next_epoch=s['epoch']+1,history=copy.deepcopy(s['history']))
    def heartbeat(self,**kw):pass
    def save_epoch(self,m,o,r,**kw):
        self.saved=copy.deepcopy(dict(model=m.state_dict(),opt=o.state_dict(),rng=r.bit_generator.state,
            torch_rng=torch.get_rng_state(),**kw))

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class Tests(unittest.TestCase):
    def test_actual_train_rng_adam_history_and_resume(self):
        torch.manual_seed(51); base=Model()
        u=np.arange(47)%7;p=np.arange(47)%11+7
        keys=np.unique(u*23+p)
        prepared={'data':dict(tr_u=u,tr_i=p,pos_key=keys,n_items=23)}
        fast,_=optimized_train(execution)
        results=[]
        for fn,split in [(execution.train,False),(fast,False),(fast,True)]:
            with tempfile.TemporaryDirectory() as d:
                m=copy.deepcopy(base);s=Store(d)
                cfg=SimpleNamespace(negative_count=1,lr=.0005,batch_size=8,epochs=2 if split else 4)
                torch.manual_seed(51)
                fn(m,prepared,cfg,'test',51,s,negative_sampler=execution.sample_uniform_negative_matrix)
                if split:
                    m=copy.deepcopy(base);s=Store(d,s.saved);cfg.epochs=4
                    fn(m,prepared,cfg,'test',51,s,negative_sampler=execution.sample_uniform_negative_matrix)
                results.append({k:v for k,v in s.saved.items() if k!='wall_clock_sec'})
        self.assertTrue(equal(results[0],results[1]))
        self.assertTrue(equal(results[1],results[2]))
    def test_worker_failure_is_propagated_and_closed(self):
        def bad(*a,**k):raise ValueError('sampling failed')
        with self.assertRaisesRegex(ValueError,'sampling failed'):
            with Batches(np.arange(3),2,np.arange(3),np.arange(3),4,None,None,bad) as b:b.take(0)
        self.assertTrue(b.pool._shutdown)

if __name__=='__main__':unittest.main()
