"""Small train-only parity tests plus sampler timing on provided data. No test metrics."""
import copy, tempfile, time, json, gc
from pathlib import Path
import numpy as np
import torch
import clv_gat_execution_fast as fast
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as ref_fixture
from test_clv_hm_gat_final8_seed49 import prepared as gat_fixture
from clv_ngcf_training_io import loss_with_tensor_logs as loss_fn


def validate(screen, prepared):
    cfg=prepared["cfg"]
    with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[0]):
        p=gat_fixture.__wrapped__(ref_fixture.__wrapped__(Path(directory)))
        p['cfg']=screen.configure(out_dir=directory)
        for label in screen.ORDER:
            torch.manual_seed(50)
            a,weights=screen.build_model(p,label)
            torch.manual_seed(50)
            b,_=screen.build_model(p,label);fast.apply(b)
            assert set(a.state_dict())==set(b.state_dict())
            oa=torch.optim.Adam(a.parameters(),lr=p['cfg'].lr)
            ob=torch.optim.Adam(b.parameters(),lr=p['cfg'].lr)
            device=next(a.parameters()).device
            u=torch.tensor([0,1,2],device=device);i=torch.tensor([1,2,3],device=device);j=torch.tensor([4,4,0],device=device)
            w=None if weights is None else torch.tensor(weights[:3],dtype=torch.float32,device=device)
            for step in range(3):
                oa.zero_grad();ob.zero_grad()
                la,_=loss_fn(a,u,i,j,w);lb,_=loss_fn(b,u,i,j,w)
                torch.testing.assert_close(la,lb,atol=2e-6,rtol=2e-5)
                la.backward();lb.backward()
                for (k,x),(kk,y) in zip(a.named_parameters(),b.named_parameters()):
                    assert k==kk
                    torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-4)
                oa.step();ob.step()
                for k,v in a.state_dict().items():
                    if v.layout==torch.strided:torch.testing.assert_close(v,b.state_dict()[k],atol=2e-6,rtol=2e-4)
            saved=Path(directory)/'resume.pt'
            torch.save({'model':b.state_dict(),'adam':ob.state_dict()},saved)
            c,_=screen.build_model(p,label);fast.apply(c)
            oc=torch.optim.Adam(c.parameters(),lr=p['cfg'].lr)
            q=torch.load(saved,weights_only=False);c.load_state_dict(q['model']);oc.load_state_dict(q['adam'])
            for k,v in b.state_dict().items():
                if v.layout==torch.strided:torch.testing.assert_close(v,c.state_dict()[k],atol=0,rtol=0)
            print('PASS',label,'3 Adam steps/all gradients/state keys/checkpoint restore',flush=True)
            del a,b,c,oa,ob,oc,q;gc.collect();torch.cuda.empty_cache()

    # Same samples AND RNG position, with varying rejection frequencies.
    for seed in range(50):
        keys=np.unique(np.array([0,1,2,10,11,20,21,22,23,24,25,26,27,28]))
        users=np.tile(np.array([0,1,2]),137);items=np.zeros_like(users)
        ra=np.random.default_rng(seed);rb=np.random.default_rng(seed)
        aa=bb=None
        try: aa=screen.sample_uniform_negative_matrix(users,items,10,keys,ra,k=1)
        except RuntimeError: pass
        try: bb=fast.sample(users,items,10,keys,rb,k=1)
        except RuntimeError: pass
        assert (aa is None)==(bb is None)
        if aa is not None: assert np.array_equal(aa,bb)
        assert ra.bit_generator.state==rb.bit_generator.state
    print('PASS negative samples and RNG states, 50 seeds',flush=True)
    d=prepared['data'];u=d['tr_u'][:cfg.batch_size];i=d['tr_i'][:cfg.batch_size]
    results=[]
    for fn in (screen.sample_uniform_negative_matrix,fast.sample):
        rng=np.random.default_rng(123);t=time.perf_counter()
        for _ in range(5):fn(u,i,d['n_items'],d['pos_key'],rng,k=1)
        results.append((time.perf_counter()-t)/5)
    print('NEGATIVE SAMPLER SECONDS',results,flush=True)
    print('ALL OPTIMIZATION CHECKS PASSED; no test evaluation or saved-run mutation',flush=True)


def validate_retention():
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder).resolve()
        class Original:
            def __init__(self):
                self.root=root/'progress'
                self.latest_checkpoint=self.root/'resume'/'test_latest.pt'
                self.latest_checkpoint.parent.mkdir(parents=True,exist_ok=True)
            def save_epoch(self,model,optimizer,rng,**state):
                if state.get('fail'):raise OSError('simulated save failure')
                self.latest_checkpoint.write_bytes(str(state['epoch']).encode())
                return self.latest_checkpoint
        fake=SimpleNamespace(ORDER=('M1','M2','M3','M4'),SafeProgressStore=Original,
                             build_model=lambda *a:None)
        p={'run_dir':root,'cfg':SimpleNamespace(epochs=300,seeds=(50,))}
        fast.install(fake,p);first=fake.SafeProgressStore;fast.install(fake,p)
        assert fake.SafeProgressStore is first
        store=fake.SafeProgressStore();store.latest_checkpoint.write_bytes(b'4')
        store.save_epoch(None,None,None,epoch=5)
        previous=store.latest_checkpoint.with_name('test_previous.pt')
        assert store.latest_checkpoint.read_bytes()==b'5' and previous.read_bytes()==b'4'
        try:store.save_epoch(None,None,None,epoch=6,fail=True)
        except OSError:pass
        else:raise AssertionError('expected failure')
        assert store.latest_checkpoint.read_bytes()==b'5' and previous.read_bytes()==b'4'
        store.save_epoch(None,None,None,epoch=6)
        assert store.latest_checkpoint.read_bytes()==b'6' and previous.read_bytes()==b'5'
        print('PASS latest/previous retention, save failure preservation, repeated installation')
