import tempfile,types,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass
import torch,numpy as np
import clv_gcn_school_fast_storage as s
@dataclass
class Id:
 stage:str='gcn';model_id:str='gcn_m1_lastweek';seed:int=51;config_hash:str='test';source_revision:str='fixed';input_hash:str='input'
class Req:
 def __init__(self,x):self.x=x
 def execute(self,**k):return self.x
class API:
 def __init__(self):self.rows=[];self.deleted=[]
 def about(self):return self
 def files(self):return self
 def get(self,**k):return Req(next(x for x in self.rows if x['id']==k['fileId']) if 'fileId' in k else {'user':{'emailAddress':s.backend.ACCOUNT}})
 def create(self,body,fields,media_body=None):
  p=Path(media_body._filename);r=dict(id=str(len(self.rows)+len(self.deleted)),name=body['name'],parents=body['parents'],ownedByMe=True,trashed=False,appProperties=body['appProperties'],size=str(p.stat().st_size),md5Checksum=hashlib.md5(p.read_bytes()).hexdigest());self.rows.append(r);return Req(r)
 def delete(self,fileId):self.rows=[r for r in self.rows if r['id']!=fileId];self.deleted.append(fileId);return Req({})
class Base:
 def __init__(self,root,ident):self.identity=ident;self.latest_checkpoint=Path(root)/'resume'/'old.pt'
 def _validate_identity(self,x):assert x==s.asdict(self.identity)
 def mark_complete(self,**state):return state
 def read_progress(self):return {'result_path':str(self.latest_checkpoint.parent/'arm.json')}
 def restore_epoch(self,*args):
  p=torch.load(self.latest_checkpoint,weights_only=False);return {'next_epoch':p['epoch']+1,'history':p['history']}
class Test(unittest.TestCase):
 def test_storage(self):
  with tempfile.TemporaryDirectory() as d:
   api=API();o=types.SimpleNamespace(SafeProgressStore=Base);ctx={'original':o,'out':Path(d)/'new','shared':{'run_dir':Path(d)/'old'}}
   with patch.object(s.backend,'listing',lambda api,q,fields=None:[{'id':'folder'}] if 'name=' in q else []),patch.object(s.backend,'rows_for',lambda api,parent,ident:sorted(api.rows,key=lambda r:int(r['appProperties']['clv_epoch']),reverse=True)),patch.object(s,'Path',side_effect=lambda p:Path(d)/'local' if p=='/content/gcn_school_fast' else Path(p)):
    s.install(ctx,api);st=o.SafeProgressStore(ctx['out']/'progress',Id());m=torch.nn.Linear(2,1);opt=torch.optim.Adam(m.parameters());rng=np.random.default_rng(51)
    for ep in range(1,31):st.save_epoch(m,opt,rng,epoch=ep,history=[{'epoch':i} for i in range(1,ep+1)],max_epoch=300)
    self.assertEqual([int(r['appProperties']['clv_epoch']) for r in api.rows],[20,30]);self.assertEqual(len(api.deleted),1);self.assertEqual(torch.load(st.previous,weights_only=False)['epoch'],29);self.assertEqual(st.restore_epoch(m,opt,rng)['next_epoch'],31)
    p=torch.load(st.latest_checkpoint,weights_only=False);p.update(epoch=300,history=[{'epoch':i} for i in range(1,301)],completed_weights_only=True);p.pop('optimizer_state');torch.save(p,st.latest_checkpoint);(st.latest_checkpoint.parent/'arm.json').write_text(s.json.dumps({'identity':s.asdict(st.identity),'epochs':300}));st.mark_complete(completed_weights_only=True)
    self.assertEqual(len(api.rows),1);self.assertEqual(api.rows[0]['appProperties']['gcn_final'],'True')
if __name__=='__main__':unittest.main()
