"""Use the established personal mounted-Drive save path; no quota/API wrapper."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import os
import shutil
import torch


def checked(path, ident):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('identity')!=asdict(ident): raise RuntimeError('Checkpoint identity mismatch')
    ep=p.get('epoch')
    if not isinstance(ep,int) or not 1<=ep<=300 or [r['epoch'] for r in p['history']]!=list(range(1,ep+1)):
        raise RuntimeError('Checkpoint history mismatch')
    for k in ('model_state','optimizer_state','numpy_rng_state','torch_rng_state','cuda_rng_state'):
        if k not in p: raise RuntimeError('Missing training state: '+k)
    return ep


def copy_verified(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(target.suffix+'.copytmp')
    with source.open('rb') as src,temp.open('wb') as dst:
        shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
    if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(temp.read_bytes()).digest():
        raise RuntimeError('Checkpoint copy hash mismatch')
    os.replace(temp,target)


def install(screen, prepared, mydrive='/content/drive/MyDrive', local_root='/content/ngcf_school_local'):
    base=getattr(screen.original,'_school_original_store',screen.original.SafeProgressStore)
    if getattr(base,'_personal_mounted',False): return
    run=Path(prepared['run_dir'])
    remote=Path(mydrive)/('ngcf_school2_checkpoints_'+prepared['config_hash'])
    if not Path(mydrive).is_dir(): raise RuntimeError('Personal Drive mount required')
    class PersonalStore(base):
        _personal_mounted=True
        def __init__(self,root,ident):
            if Path(root)!=run/'progress' or ident.seed not in (50,51): raise RuntimeError('Wrong run/seed')
            super().__init__(root,ident)
        def restore_epoch(self,model,optimizer,rng):
            local=Path(local_root)/self.identity.config_hash/self.latest_checkpoint.name
            candidates=[p for p in (self.latest_checkpoint,local) if p.exists()]
            candidates+=list(remote.glob(f'{self.identity.model_id}_s{self.identity.seed}_ep*.pt'))
            ranked=sorted([(checked(p,self.identity),str(p),p) for p in candidates],reverse=True)
            if ranked:
                ep,_,latest=ranked[0]
                previous=next((p for e,_,p in ranked if e<ep),None)
                prevpath=self.latest_checkpoint.with_suffix('.previous.pt')
                if previous is not None and not prevpath.exists(): copy_verified(previous,prevpath)
                if latest!=self.latest_checkpoint: copy_verified(latest,self.latest_checkpoint)
                print(f'[개인 Drive 복원] epoch {ep}; API 저장·용량조회 제약 없음',flush=True)
            return super().restore_epoch(model,optimizer,rng)
        def save_epoch(self,model,optimizer,rng,**state):
            if self.latest_checkpoint.exists():
                old=checked(self.latest_checkpoint,self.identity)
                if state['epoch']<=old: raise RuntimeError('Refuse checkpoint rollback')
                copy_verified(self.latest_checkpoint,self.latest_checkpoint.with_suffix('.previous.pt'))
            result=super().save_epoch(model,optimizer,rng,**state)
            print(f'[개인 Drive 저장] epoch {state["epoch"]}: 최신·직전 유지',flush=True)
            return result
    screen.original.SafeProgressStore=PersonalStore
    print('기존 개인 NGCF 방식: 마운트 Drive 매epoch 저장. 학교 API/고정여유/재시도 wrapper 해제.',flush=True)
