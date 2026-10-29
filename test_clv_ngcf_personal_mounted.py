from dataclasses import dataclass,asdict
from types import SimpleNamespace
import torch
import pytest
import clv_ngcf_personal_mounted as m

@dataclass
class Id:
    config_hash:str='run'
    model_id:str='ngcf_m5_b_lastweek'
    seed:int=50

def payload(ep):
    return dict(identity=asdict(Id()),epoch=ep,history=[{'epoch':i} for i in range(1,ep+1)],
        model_state={},optimizer_state={},numpy_rng_state={},torch_rng_state=[],cuda_rng_state=[])

def test_resume_newest_and_keep_previous(tmp_path):
    root=tmp_path/'run'; drive=tmp_path/'drive';drive.mkdir();local=tmp_path/'local'
    class Base:
        def __init__(self,root,ident):
            self.identity=ident;self.latest_checkpoint=root/'latest.pt';root.mkdir(parents=True,exist_ok=True)
        def restore_epoch(self,*a):return {'next_epoch':m.checked(self.latest_checkpoint,self.identity)+1}
        def save_epoch(self,*a,**s):torch.save(payload(s['epoch']),self.latest_checkpoint)
    screen=SimpleNamespace(original=SimpleNamespace(SafeProgressStore=Base,_school_original_store=Base))
    remote=drive/'ngcf_school2_checkpoints_run';remote.mkdir()
    torch.save(payload(254),remote/'ngcf_m5_b_lastweek_s50_ep254.pt')
    (local/'run').mkdir(parents=True);torch.save(payload(255),local/'run/latest.pt')
    m.install(screen,{'run_dir':root,'config_hash':'run'},drive,local)
    store=screen.original.SafeProgressStore(root/'progress',Id())
    assert store.restore_epoch(None,None,None)['next_epoch']==256
    assert m.checked(store.latest_checkpoint.with_suffix('.previous.pt'),Id())==254
    store.save_epoch(None,None,None,epoch=256)
    assert m.checked(store.latest_checkpoint,Id())==256
    assert m.checked(store.latest_checkpoint.with_suffix('.previous.pt'),Id())==255
    assert m.checked(remote/'ngcf_m5_b_lastweek_s50_ep254.pt',Id())==254
    with pytest.raises(RuntimeError,match='rollback'):store.save_epoch(None,None,None,epoch=254)
