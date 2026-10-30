"""M5 combination parity, resumability and failed-publish preservation."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import clv_gat_execution_fast as fast
import clv_hm_gat_m5_seed50 as screen
from clv_ngcf_training_io import loss_with_tensor_logs
from test_clv_hm_gat_m5_seed50 import prepared, gs_fixture, reference_fixture


@pytest.mark.parametrize('label', screen.ORDER)
def test_m5_loss_gradients_adam_and_resume(prepared, label):
    torch.manual_seed(50)
    original, weights = screen.build_model(prepared, label)
    optimized = copy.deepcopy(original)
    keys = set(optimized.state_dict())
    fast.apply(optimized)
    assert set(optimized.state_dict()) == keys
    opts = [torch.optim.Adam(m.parameters(), lr=prepared['cfg'].lr)
            for m in (original, optimized)]
    u,i,j = torch.tensor([0,1,2]),torch.tensor([1,2,3]),torch.tensor([4,4,0])
    w = None if weights is None else torch.tensor(weights[:3], dtype=torch.float32)
    for step in range(3):
        losses=[]
        for m,opt in zip((original, optimized),opts):
            opt.zero_grad();loss,_=loss_with_tensor_logs(m,u,i,j,w)
            loss.backward();losses.append(loss.detach())
        torch.testing.assert_close(*losses,atol=2e-6,rtol=2e-5)
        for (name,a),(other,b) in zip(original.named_parameters(),optimized.named_parameters()):
            assert name==other
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-4)
        for opt in opts:opt.step()
        for a,b in zip(original.parameters(),optimized.parameters()):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
    checkpoint=prepared['run_dir']/'execution_test.pt'
    torch.save({'model':optimized.state_dict(),'optimizer':opts[1].state_dict()},checkpoint)
    restored,_=screen.build_model(prepared,label);fast.apply(restored)
    state=torch.load(checkpoint,weights_only=False)
    restored.load_state_dict(state['model'])
    opt=torch.optim.Adam(restored.parameters(),lr=prepared['cfg'].lr)
    opt.load_state_dict(state['optimizer'])
    # One more update after roundtrip must match the uninterrupted optimized run.
    for m,o in ((optimized,opts[1]),(restored,opt)):
        o.zero_grad();loss,_=loss_with_tensor_logs(m,u,i,j,w);loss.backward();o.step()
    for a,b in zip(optimized.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_sampler_rng_and_rejection_limit():
    keys=np.array([0,1,2,10,11,20,21,22,23,24,25,26,27,28])
    users=np.tile([0,1,2],137); positives=np.zeros_like(users)
    for seed in range(50):
        a=np.random.default_rng(seed);b=np.random.default_rng(seed)
        outcomes=[]
        for fn,rng in ((screen.sample_uniform_negative_matrix,a),(fast.sample,b)):
            try:outcomes.append(fn(users,positives,10,keys,rng,k=1))
            except RuntimeError:outcomes.append(None)
        assert (outcomes[0] is None)==(outcomes[1] is None)
        if outcomes[0] is not None:np.testing.assert_array_equal(*outcomes)
        assert a.bit_generator.state==b.bit_generator.state


@pytest.mark.parametrize('failure', ['before_publish','after_publish'])
def test_m5_install_and_preserve_failed_save(tmp_path, monkeypatch, failure):
    root=tmp_path.resolve()
    recovery=root/'local_recovery';recovery.mkdir()
    monkeypatch.setattr('tempfile.mkdtemp',lambda **kw:str(recovery))
    class Original:
        def __init__(self):
            self.root=root/'progress';self.latest_checkpoint=self.root/'resume'/'model_latest.pt'
            self.latest_checkpoint.parent.mkdir(parents=True)
        def save_epoch(self,*a,**kw):
            if failure=='before_publish':raise OSError('before')
            self.latest_checkpoint.write_bytes(b'5')
            raise OSError('metadata failed after durable publish')
    runner=SimpleNamespace(ORDER=screen.ORDER,SafeProgressStore=Original,build_model=lambda *a:None)
    p={'run_dir':root,'cfg':SimpleNamespace(epochs=300,seeds=(50,))}
    fast.install(runner,p);cls=runner.SafeProgressStore;fast.install(runner,p)
    assert cls is runner.SafeProgressStore
    store=cls();store.latest_checkpoint.write_bytes(b'4')
    previous=store.latest_checkpoint.with_name('model_previous.pt');previous.write_bytes(b'3')
    with pytest.raises(OSError):store.save_epoch(None,None,None,epoch=5)
    assert (recovery/'previous.pt').read_bytes()==b'4'
    assert previous.read_bytes()==b'3'
    assert store.latest_checkpoint.read_bytes()==(b'4' if failure=='before_publish' else b'5')
