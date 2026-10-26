import sys,tempfile,types,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import clv_drive_checkpoint_retention as m
_old_torch=sys.modules.get('torch')
sys.modules['torch']=types.SimpleNamespace()
events=[]
class Request:
 def execute(self): return {'user':{'emailAddress':'jungunb@u.sogang.ac.kr'}}
api=types.SimpleNamespace(about=lambda:types.SimpleNamespace(get=lambda **kw:Request()))
class Store:
 def save_epoch(self,*a,**kw): events.append('save')
screen=types.SimpleNamespace(SafeProgressStore=Store)
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d);prepared={'run_dir':p}
 m.prune=lambda *a,**kw:events.append('old')
 m.install(screen,api,prepared)
 # Reload into a new module, as the user's installed preservation cell does.
 new=types.ModuleType('new_retention');exec(pathlib.Path(m.__file__).read_text(),new.__dict__)
 new.prune=lambda *a,**kw:events.append('new')
 new.install(screen,api,prepared)
 s=Store();s.root=p/'progress';s.latest_checkpoint=p/'x_latest.pt';s.latest_checkpoint.write_bytes(b'x')
 s.save_epoch()
 assert events==['new','save','new'],events
 print('PASS: reinstallation replaces old cleanup closure, saves exactly once')

if _old_torch is None: sys.modules.pop('torch',None)
else: sys.modules['torch']=_old_torch
