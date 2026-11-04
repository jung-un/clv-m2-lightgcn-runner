"""Read-only experiment probe on an idle prepared L4; no test evaluation or run writes."""
import copy
import gc
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from clv_gcn51_execution_prefetch import optimized_train

class ProbeStore:
    def __init__(self,root):
        self.root=Path(root); (self.root/'timing').mkdir();self.saved=None
    def restore_epoch(self,*args):return None
    def heartbeat(self,**kw):pass
    def save_epoch(self,model,opt,rng,**state):
        # Only final state captured; no production checkpoint overwritten.
        if state['epoch']==4:
            self.saved={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k!='adj'},
                'optimizer':copy.deepcopy(opt.state_dict()),'rng':copy.deepcopy(rng.bit_generator.state),
                'torch_rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'history':copy.deepcopy(state['history'])}
            for item in self.saved['optimizer']['state'].values():
                for k,v in item.items():
                    if torch.is_tensor(v):item[k]=v.detach().cpu().clone()

def compare(a,b):
    if torch.is_tensor(a):
        if a.is_floating_point():torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)

def run(screen,prepared):
    assert 'L4' in torch.cuda.get_device_name(0)
    rng_cpu=torch.get_rng_state();rng_gpu=torch.cuda.get_rng_state_all();rng_np=np.random.get_state()
    original=getattr(screen.execution,'_before_gcn51_prefetch',screen.execution.train)
    fast,digest=optimized_train(screen.execution)
    probe=dict(prepared); data=dict(prepared['data']);probe['data']=data
    n=min(len(data['tr_u']),prepared['cfg'].batch_size*12)
    data['tr_u']=data['tr_u'][:n];data['tr_i']=data['tr_i'][:n]
    cfg=SimpleNamespace(**vars(prepared['cfg']));cfg.epochs=4
    results=[]
    try:
        for name,fn in [('baseline',original),('prefetch',fast)]:
            screen.base.v3.set_seed(51)
            model,_=screen.build_model(prepared,'M2')
            screen.base.v3.set_seed(51)
            with tempfile.TemporaryDirectory(prefix='gcn51-probe-') as root:
                store=ProbeStore(root)
                fn(model,probe,cfg,'probe_m2',51,store,negative_sampler=screen.sample_uniform_negative_matrix)
                timing=pd.read_csv(Path(root)/'timing/probe_m2_s51.csv')
                results.append((store.saved,float(timing.iloc[-1].train_seconds)))
                print('PROBE',name,'12_batch_seconds',results[-1][1],flush=True)
            del model,store;gc.collect();torch.cuda.empty_cache()
        compare(results[0][0],results[1][0])
        result={'exact_model_adam_rng_history':True,'baseline_seconds':results[0][1],
                'prefetch_seconds':results[1][1],'speed_ratio':results[0][1]/results[1][1],
                'training_source_sha256':digest,'probe_only':True,'test_evaluated':False}
        print('PROBE_VERIFIED',result,flush=True)
        return result
    finally:
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_gpu);np.random.set_state(rng_np)
        gc.collect();torch.cuda.empty_cache()
