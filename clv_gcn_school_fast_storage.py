"""GCN-only local epoch saves; immutable Drive publication every ten epochs."""
from dataclasses import asdict
from pathlib import Path
import hashlib, json, os, shutil, time
import torch
import clv_ngcf_school_storage as backend


def install(ctx, api):
    o = ctx['original']
    if getattr(o.SafeProgressStore, '_gcn_fast', False):
        return
    info = backend.execute(api.about().get(fields='user(emailAddress)'))
    if info['user']['emailAddress'] != backend.ACCOUNT:
        raise RuntimeError('학교 계정 인증 필요')
    folder_name = 'dh_gcn_remaining_school_drive10_v1'
    rows = backend.listing(api, "'me' in owners and trashed=false and name='"+folder_name+"'")
    if len(rows)>1: raise RuntimeError('중복 checkpoint 폴더')
    parent = rows[0]['id'] if rows else api.files().create(body={'name':folder_name,'mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']
    cache = {'time':0,'free':0}
    def guard(required=0):
        if time.monotonic()-cache['time']>600:
            rows=backend.listing(api,"'me' in owners",'id,mimeType,quotaBytesUsed')
            used=sum(int(r['quotaBytesUsed']) for r in rows if r['mimeType'] not in ('application/vnd.google-apps.folder','application/vnd.google-apps.shortcut'))
            cache.update(time=time.monotonic(),free=10_000_000_000-used)
        if cache['free'] < max(2_000_000_000,required):
            raise RuntimeError('총 여유 2GB 미만: 로컬/Drive checkpoint 보존 후 중단')
    guard()
    base=o.SafeProgressStore
    roots={str(Path(ctx['out']).resolve()/'progress'),str(Path(ctx['shared']['run_dir']).resolve()/'progress')}
    def digest(path):
        h=hashlib.md5()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()
    def entries(ident):
        return backend.rows_for(api,parent,ident)
    def publish(store,final=False):
        from googleapiclient.http import MediaFileUpload
        path=store.latest_checkpoint; payload=torch.load(path,map_location='cpu',weights_only=False)
        store._validate_identity(payload['identity']); epoch=payload['epoch']; md5=digest(path)
        if [r['epoch'] for r in payload['history']] != list(range(1,epoch+1)):raise RuntimeError('checkpoint 이력 불일치')
        if final:
            if epoch!=300 or not payload.get('completed_weights_only'):raise RuntimeError('미완료 최종본')
            progress=store.read_progress(); row=json.loads(Path(progress['result_path']).read_text())
            if row['identity']!=payload['identity'] or row['epochs']!=300:raise RuntimeError('완료 결과 불일치')
        prior=entries(store.identity)
        match=[r for r in prior if r['md5Checksum']==md5 and int(r['size'])==path.stat().st_size]
        if match: current=match[0]
        else:
            guard(path.stat().st_size)
            props={'clv_split_run':store.identity.config_hash,'clv_model':store.identity.model_id,'clv_seed':str(store.identity.seed),'clv_epoch':str(epoch),'clv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gcn_final':str(final)}
            current=api.files().create(body={'name':path.stem+f'_ep{epoch}_'+md5[:12]+'.pt','parents':[parent],'appProperties':props},media_body=MediaFileUpload(str(path),resumable=True),fields=backend.FIELDS).execute(num_retries=3)
            if current.get('md5Checksum')!=md5 or int(current['size'])!=path.stat().st_size:raise RuntimeError('Drive 저장 검증 실패; 기존본 보존')
            cache['free']-=path.stat().st_size
        # Only our dedicated folder's same identity files; never legacy/other runs.
        all_rows=[current]+[r for r in prior if r['id']!=current['id']]
        all_rows.sort(key=lambda r:(int(r['appProperties']['clv_epoch']),r['id']==current['id']),reverse=True)
        keep={r['id'] for r in all_rows[:1 if final else 2]}
        for r in all_rows:
            if r['id'] in keep:continue
            fresh=backend.execute(api.files().get(fileId=r['id'],fields=backend.FIELDS))
            if any(fresh.get(k)!=r.get(k) for k in ('size','md5Checksum','appProperties','parents')):raise RuntimeError('이전본 동시 변경')
            backend.execute(api.files().delete(fileId=r['id']))
            cache['free']+=int(r['size'])
        print(f'[Drive 저장] epoch {epoch}'+(' 최종 가중치' if final else ' / 최신·직전 유지'),flush=True)
    class Store(base):
        _gcn_fast=True
        def __init__(self,root,ident):
            if str(Path(root).resolve()) not in roots:raise RuntimeError('GCN 저장 범위 밖')
            super().__init__(root,ident)
            self.legacy=self.latest_checkpoint
            self.legacy_previous=self.legacy.with_name(self.legacy.stem.replace('_latest','_previous')+'.pt')
            local=Path('/content/gcn_school_fast')/hashlib.sha256(json.dumps(asdict(ident),sort_keys=True).encode()).hexdigest()
            local.mkdir(parents=True,exist_ok=True)
            self.latest_checkpoint=local/'latest.pt'; self.previous=local/'previous.pt'
        def restore_epoch(self,model,optimizer,rng):
            from googleapiclient.http import MediaIoBaseDownload
            candidates=[]
            for path in (self.latest_checkpoint,self.previous,self.legacy,self.legacy_previous):
                if path.exists():
                    p=torch.load(path,map_location='cpu',weights_only=False);self._validate_identity(p['identity']); candidates.append((p['epoch'],path))
            rows=entries(self.identity)
            if rows and int(rows[0]['appProperties']['clv_epoch'])>max([x[0] for x in candidates],default=0):
                r=rows[0]; temp=self.latest_checkpoint.with_suffix('.download')
                with temp.open('wb') as f:
                    dl=MediaIoBaseDownload(f,api.files().get_media(fileId=r['id']));done=False
                    while not done: _,done=dl.next_chunk(num_retries=3)
                if digest(temp)!=r['md5Checksum']:raise RuntimeError('복구 해시 불일치')
                p=torch.load(temp,map_location='cpu',weights_only=False);self._validate_identity(p['identity']);candidates.append((p['epoch'],temp))
            if candidates:
                _,path=max(candidates,key=lambda x:x[0])
                if path!=self.latest_checkpoint:shutil.copy2(path,self.latest_checkpoint)
            restored=super().restore_epoch(model,optimizer,rng)
            if restored and restored['next_epoch']==301:publish(self)
            return restored
        def save_epoch(self,model,optimizer,rng,**state):
            begin=time.perf_counter()
            payload=dict(identity=asdict(self.identity),model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],**state)
            temp=self.latest_checkpoint.with_suffix('.tmp')
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            if self.latest_checkpoint.exists():os.replace(self.latest_checkpoint,self.previous)
            os.replace(temp,self.latest_checkpoint)
            local=time.perf_counter()-begin
            if state['epoch']%10==0 or state['epoch']==300:publish(self)
            self.last_save_timing={'checkpoint_local_seconds':local,'checkpoint_drive_seconds':time.perf_counter()-begin-local,'checkpoint_metadata_seconds':0,'checkpoint_bytes':self.latest_checkpoint.stat().st_size}
            return self.latest_checkpoint
        def mark_complete(self,**state):
            result=super().mark_complete(**state)
            if state.get('completed_weights_only'):publish(self,final=True)
            return result
    o.SafeProgressStore=Store
    o.STORAGE_GUARD=None
    print('GCN: 로컬 매epoch, Drive 10epoch/최종; 총여유2GB; 용량조회10분 캐시. H&M 변경 없음')
