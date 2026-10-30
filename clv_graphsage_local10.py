"""Storage-only local-first cadence for personal H&M GraphSAGE seed50."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import torch


def limited(exc):
    status = getattr(getattr(exc, 'resp', None), 'status', None)
    try:
        reasons = {x.get('reason') for x in json.loads(exc.content).get('error', {}).get('errors', [])}
    except (AttributeError, ValueError, TypeError):
        reasons = set()
    return status == 429 or (status == 403 and bool(reasons & {'rateLimitExceeded', 'userRateLimitExceeded'}))


def validate(path, identity):
    p = torch.load(path, map_location='cpu', weights_only=False)
    e = p.get('epoch')
    if p.get('identity') != asdict(identity) or not isinstance(e, int) or not 1 <= e <= 300:
        raise RuntimeError('checkpoint identity/epoch mismatch')
    if [r.get('epoch') for r in p.get('history', [])] != list(range(1, e + 1)):
        raise RuntimeError('checkpoint history mismatch')
    for k in ('model_state','optimizer_state','numpy_rng_state','torch_rng_state','cuda_rng_state'):
        if k not in p: raise RuntimeError('missing checkpoint state: '+k)
    return p


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def install(screen, api, prepared, *, local_root='/content/graphsage_seed50_core_checkpoints'):
    import clv_personal_graphsage_checkpoint as personal
    screen.validate_config(prepared['cfg'])
    assert prepared['cfg'].seeds == (50,) and screen.ORDER == ('M1','M2','M3','M4')
    run=Path(prepared['run_dir'])
    record=json.loads((run/'personal_checkpoint_storage.json').read_text())
    assert record['account']==personal.ACCOUNT
    parent=record['folder_id']
    backend=personal.backend()
    # No blocking API retry loop: explicit quota errors defer to a later save boundary.
    backend.execute=lambda request:request.execute()
    original=screen._personal_original_store
    remote_class=screen.SafeProgressStore
    if getattr(remote_class,'_local10',False):
        print('이미 로컬 매epoch / Drive10epoch 적용됨'); return
    pending={}
    retry_after=[0.]
    local=Path(local_root)/prepared['config_hash']
    local.mkdir(parents=True,exist_ok=True)

    def flush():
        if time.monotonic()<retry_after[0]: return
        for key,(path,ident,epoch) in list(pending.items()):
            try:
                backend.publish(api,parent,path,ident,epoch)
            except Exception as exc:
                if not limited(exc): raise
                retry_after[0]=time.monotonic()+600
                print(f'[Drive 보류] API 제한; 로컬 epoch{epoch} 보존. 학습 계속, 10분 이후 저장 경계에서 재시도.',flush=True)
                return
            pending.pop(key)
            print(f'[Drive 저장 확인] {ident.model_id} epoch{epoch}; 원격 최신·직전 유지',flush=True)

    class Local10(original):
        _personal_storage_installed=True
        _local10=True
        def __init__(self,root,ident):
            assert Path(root).resolve()==(run/'progress').resolve()
            assert ident in [screen.identity(prepared,m) for m in screen.ORDER]
            super().__init__(root,ident)
            self.latest_checkpoint=local/self.latest_checkpoint.name
            self.previous=self.latest_checkpoint.with_name(self.latest_checkpoint.stem+'_previous.pt')
        def restore_epoch(self,model,optimizer,rng):
            candidates=[]
            for path in (self.latest_checkpoint,self.previous):
                if path.exists():
                    p=validate(path,self.identity); candidates.append((p['epoch'],path)); del p
            if candidates:
                e,path=max(candidates,key=lambda v:v[0])
                normal=self.latest_checkpoint
                self.latest_checkpoint=path
                try: result=original.restore_epoch(self,model,optimizer,rng)
                finally:self.latest_checkpoint=normal
                print(f'[검증된 로컬 재개] epoch{e}; 원격 파일로 덮어쓰지 않음',flush=True)
                return result
            # A fresh runtime must restore and validate remote state; do not start over on API failure.
            remote=remote_class(self.root,self.identity)
            result=remote.restore_epoch(model,optimizer,rng)
            if remote.latest_checkpoint.exists():
                validate(remote.latest_checkpoint,self.identity)
                if remote.latest_checkpoint!=self.latest_checkpoint:
                    shutil.copyfile(remote.latest_checkpoint,self.latest_checkpoint)
            return result
        def save_epoch(self,model,optimizer,rng,**state):
            begin=time.perf_counter()
            from clv_run_state import clone_state
            payload=dict(identity=asdict(self.identity),model_state=clone_state(model),
                optimizer_state=optimizer.state_dict(),numpy_rng_state=rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],**state)
            temp=self.latest_checkpoint.with_suffix('.local10.tmp')
            with temp.open('wb') as f:
                torch.save(payload,f); f.flush(); os.fsync(f.fileno())
            if self.latest_checkpoint.exists(): os.replace(self.latest_checkpoint,self.previous)
            os.replace(temp,self.latest_checkpoint)
            saved=time.perf_counter()
            e=int(state['epoch'])
            if e%10==0 or e==300:
                pending[self.identity.model_id]=(self.latest_checkpoint,self.identity,e)
                flush()
            # Keep local/remote status separate; original training uses this timing dictionary only.
            (local/(self.identity.model_id+'_sync.json')).write_text(json.dumps({'local_epoch':e,
                'pending_models':list(pending),'drive_interval':10}))
            self.last_save_timing=dict(checkpoint_local_seconds=saved-begin,
                checkpoint_drive_seconds=time.perf_counter()-saved,checkpoint_metadata_seconds=0.,
                checkpoint_bytes=self.latest_checkpoint.stat().st_size)
            return self.latest_checkpoint
    screen.SafeProgressStore=Local10
    screen.flush_pending_checkpoints=flush
    screen.pending_checkpoints=pending
    print('적용: 로컬 매epoch 최신·직전 / Drive10epoch+최종 / API 제한은 보류 후 학습 계속. 원격 미반영 구간은 런타임 소멸 시 복구 불가.',flush=True)
