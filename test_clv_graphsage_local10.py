import sys, types, pickle, tempfile, json, unittest, importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from unittest.mock import patch

@dataclass
class Identity:
    model_id:str='m1'
    seed:int=50
    config_hash:str='run'

class Tests(unittest.TestCase):
    def test_quota_continues_cadence_and_resume(self):
        fake=types.SimpleNamespace(save=lambda p,f:pickle.dump(p,f),load=lambda p,**kw:pickle.loads(Path(p).read_bytes()),get_rng_state=lambda:b'rng',cuda=types.SimpleNamespace(is_available=lambda:False))
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,{'torch':fake,'clv_run_state':types.SimpleNamespace(clone_state=lambda m:{'w':1})}):
            spec=importlib.util.spec_from_file_location('subject','clv_graphsage_local10.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            root=Path(d); (root/'personal_checkpoint_storage.json').write_text(json.dumps({'account':'personal','folder_id':'folder'}))
            class Base:
                def __init__(self,r,i): self.root=r; self.identity=i; self.latest_checkpoint=Path(r)/'latest.pt'
                def restore_epoch(self,*a): return {'next_epoch':mod.validate(self.latest_checkpoint,self.identity)['epoch']+1}
            attempts=[]
            class Quota(Exception):
                resp=types.SimpleNamespace(status=403); content=b'{"error":{"errors":[{"reason":"userRateLimitExceeded"}]}}'
            backend=types.SimpleNamespace(publish=lambda *a:(attempts.append(a[-1]), (_ for _ in ()).throw(Quota()))[-1])
            personal=types.SimpleNamespace(ACCOUNT='personal',backend=lambda:backend)
            screen=types.SimpleNamespace(validate_config=lambda c:None,ORDER=('M1','M2','M3','M4'),_personal_original_store=Base,SafeProgressStore=Base,identity=lambda p,m:Identity())
            prepared={'cfg':types.SimpleNamespace(seeds=(50,)),'config_hash':'run','run_dir':root}
            with patch.dict(sys.modules,{'clv_personal_graphsage_checkpoint':personal}):mod.install(screen,None,prepared,local_root=str(root/'local'))
            store=screen.SafeProgressStore(root/'progress',Identity()); rng=types.SimpleNamespace(bit_generator=types.SimpleNamespace(state={'x':1})); opt=types.SimpleNamespace(state_dict=lambda:{'adam':1})
            for e in range(1,12):store.save_epoch(None,opt,rng,epoch=e,history=[{'epoch':n} for n in range(1,e+1)])
            self.assertEqual(attempts,[10]);self.assertEqual(mod.validate(store.latest_checkpoint,Identity())['epoch'],11);self.assertEqual(mod.validate(store.previous,Identity())['epoch'],10)
            self.assertEqual(store.restore_epoch(None,None,None)['next_epoch'],12)
            self.assertFalse(mod.limited(RuntimeError('corruption')))
            self.assertFalse(mod.limited(types.SimpleNamespace(resp=types.SimpleNamespace(status=403),content=b'{"error":{"errors":[{"reason":"forbidden"}]}}')))

if __name__=='__main__':unittest.main()
