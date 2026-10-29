from types import SimpleNamespace
import numpy as np
import pytest
import torch
from clv_run_state import ProgressStore, RunIdentity
from clv_hm_checkpoint_cadence import install


def setup(tmp_path):
    class Store(ProgressStore):
        _retention_installed = True
        calls = []
        fail = False
        def save_epoch(self, *a, **kw):
            self.calls.append(kw['epoch'])
            if self.fail:
                raise OSError('Drive unavailable')
            return super().save_epoch(*a, **kw)
    p={'run_dir':tmp_path/'run'}
    install(SimpleNamespace(SafeProgressStore=Store),p,local_root=tmp_path/'local')
    s=Store(p['run_dir']/'progress',RunIdentity('hm','graphsage_m5_a_lastweek',49,'hash','source','input'))
    m=torch.nn.Linear(2,1); o=torch.optim.Adam(m.parameters()); r=np.random.default_rng(49)
    return s,m,o,r


def save(s,m,o,r,e):
    return s.save_epoch(m,o,r,epoch=e,history=[{'epoch':i} for i in range(1,e+1)],max_epoch=300)


def test_local_every_epoch_drive_five_and_exact_restore(tmp_path):
    s,m,o,r=setup(tmp_path)
    for e in range(1,8):
        o.zero_grad(); m(torch.ones(1,2)).sum().backward();o.step();r.random()
        p=save(s,m,o,r,e)
    assert s.calls==[5]
    assert torch.load(s.latest_checkpoint,weights_only=False)['epoch']==5
    local=torch.load(p,weights_only=False)
    assert local['epoch']==7
    assert torch.load(p.with_name('previous.pt'),weights_only=False)['epoch']==6
    expected=r.random(); rngstate=torch.get_rng_state().clone()
    with torch.no_grad():
        for v in m.parameters():v.zero_()
    out=s.restore_epoch(m,o,r)
    assert out['next_epoch']==8 and r.random()==expected
    assert torch.equal(torch.get_rng_state(),rngstate)
    assert all(torch.equal(v,local['model_state'][k]) for k,v in m.state_dict().items())
    assert o.state_dict()['param_groups']==local['optimizer_state']['param_groups']


def test_failed_final_publish_retried_before_evaluation(tmp_path):
    s,m,o,r=setup(tmp_path)
    save(s,m,o,r,295)
    s.fail=True
    with pytest.raises(OSError):save(s,m,o,r,300)
    assert torch.load(s.latest_checkpoint,weights_only=False)['epoch']==295
    s.fail=False
    assert s.restore_epoch(m,o,r)['next_epoch']==301
    assert s.calls==[295,300,300]
    assert torch.load(s.latest_checkpoint,weights_only=False)['epoch']==300


def test_new_runtime_uses_drive_and_preserves_identity(tmp_path):
    s,m,o,r=setup(tmp_path)
    save(s,m,o,r,5)
    assert s.restore_epoch(m,o,r)['next_epoch']==6
    other=s.__class__(s.root,RunIdentity('hm','graphsage_m5_a_lastweek',50,'hash','source','input'))
    other.latest_checkpoint=s.latest_checkpoint
    with pytest.raises(RuntimeError):other.restore_epoch(m,o,r)
