"""Verify the real pinned checkpoint implementations using a tiny CPU model."""
from pathlib import Path
import sys,subprocess
import argparse
parser = argparse.ArgumentParser(description='CPU synthetic checkpoint equality; never runs research training/test.')
parser.add_argument('job', choices=['dh-gat', 'dh-ngcf', 'hm-ngcf', 'hm-lightgcn'])
parser.add_argument('--root', type=Path, default=Path('/workspace/clv'))
args = parser.parse_args()
ROOT = args.root.resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as s
job = args.job
screen,cfg=s.load_runner(job,ROOT)
import torch,numpy as np
from unittest.mock import patch
from tempfile import TemporaryDirectory
if job=='hm-lightgcn':screen.load_references(cfg)
elif job=='hm-ngcf':screen.base.load_references(screen.base.configure(dataset='hm',seed=49,out_dir=cfg.out_dir,m2_reference=cfg.m2_reference,m4_reference=cfg.m4_reference))
owner=screen.original if job in ('dh-ngcf','dh-gat') else screen
key='ProgressStore' if job=='hm-lightgcn' else 'SafeProgressStore'
original=getattr(owner,key)
with TemporaryDirectory() as tmp:
 root=Path(tmp)
 with patch.object(s,'guard_space'):
  s.install_storage(screen,job,root)
  wrapper=getattr(owner,key)
  ident=screen.base.RunIdentity('runpod-test','model',42,'config','revision','input')
  model=torch.nn.Linear(3,2);optimizer=torch.optim.Adam(model.parameters());rng=np.random.default_rng(42)
  loss=model(torch.ones(1,3)).sum();loss.backward();optimizer.step()
  a=original(root/'original',ident);b=wrapper(root/'wrapped',ident)
  for epoch in [1,2]:
   history=[{'epoch':i,'loss':1.0} for i in range(1,epoch+1)]
   state=dict(epoch=epoch,history=history,selection='none',max_epoch=300)
   a.save_epoch(model,optimizer,rng,**state);b.save_epoch(model,optimizer,rng,**state)
  p=torch.load(a.latest_checkpoint,weights_only=False);q=torch.load(b.latest_checkpoint,weights_only=False)
  def same(x,y):
   if torch.is_tensor(x):assert torch.equal(x,y)
   elif isinstance(x,dict):
    assert x.keys()==y.keys()
    for k in x:same(x[k],y[k])
   elif isinstance(x,(list,tuple)):
    assert len(x)==len(y)
    for a,b in zip(x,y):same(a,b)
   else:assert x==y,(x,y)
  same(p,q)
  prev=torch.load(b.latest_checkpoint.with_suffix('.previous.pt'),weights_only=False)
  assert prev['epoch']==1
  restored_model=torch.nn.Linear(3,2);restored_optimizer=torch.optim.Adam(restored_model.parameters());rr=np.random.default_rng(999)
  res=b.restore_epoch(restored_model,restored_optimizer,rr)
  assert res['next_epoch']==3
  same(model.state_dict(),restored_model.state_dict());same(optimizer.state_dict(),restored_optimizer.state_dict());same(rng.bit_generator.state,rr.bit_generator.state)
print('PASS',job,'pinned import/config/references, original/wrapped state equality, previous epoch, restored model/Adam/RNG')
