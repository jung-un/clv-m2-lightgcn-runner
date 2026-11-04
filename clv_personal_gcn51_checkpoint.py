"""Bounded personal Drive storage for the NEW H&M GCN seed51 partition run.

Reuse the verified immutable-upload/hash/retain-two implementation in an
isolated module instance. Never change another runner's storage globals.
Only old epoch files created inside this new run's dedicated folder are removed.
"""
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import time
import os

ACCOUNT = 'azsabae@gmail.com'
MIN_FREE = 5_000_000_000
START_FREE = 15_000_000_000


def backend():
    spec = importlib.util.spec_from_file_location(
        '_gcn51_personal_storage_backend', Path(__file__).with_name('clv_ngcf_school_storage.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def check_space(api, required=0, initial=False):
        info = module.execute(api.about().get(fields='user(emailAddress),storageQuota'))
        if info['user']['emailAddress'] != ACCOUNT:
            raise RuntimeError('개인 Drive azsabae@gmail.com으로 인증하세요. 학교 계정 사용 금지.')
        quota = info.get('storageQuota', {})
        if 'limit' not in quota or 'usage' not in quota:
            raise RuntimeError('개인 Drive 실제 여유 용량을 확인할 수 없습니다')
        available = int(quota['limit']) - int(quota['usage'])
        needed = max(START_FREE if initial else MIN_FREE, MIN_FREE + int(required))
        if available < needed:
            raise RuntimeError(f'개인 Drive 여유 {available/1e9:.2f}GB < 필요 {needed/1e9:.2f}GB. 기존 파일 보존.')
        return available

    # about.storageQuota is ONE request; never enumerate the whole Drive.
    module.check_space = check_space
    return module


def install(screen, api, prepared, *, local_root='/content/gcn_seed51_checkpoints', drive_interval=1):
    import torch
    if type(drive_interval) is not int or drive_interval not in (1, 10):
        raise ValueError('Drive 저장 간격은 1 또는 10epoch만 허용합니다')
    screen.validate_config(prepared['cfg'])
    if prepared['cfg'].seeds != (51,) or len(screen.ORDER) != 8:
        raise RuntimeError('개인 GCN seed51 범위 밖')
    storage = backend()
    free = storage.check_space(api, initial=True)
    run_hash = prepared['config_hash']
    name = 'hm_gcn_seed51_checkpoints_' + run_hash
    query = ("'me' in owners and trashed = false and name = '" + name +
             "' and mimeType = 'application/vnd.google-apps.folder'")
    rows = storage.listing(api, query)
    if len(rows) > 1:
        raise RuntimeError('동일 체크포인트 폴더 중복: 자동 선택하지 않습니다')
    if rows:
        row = rows[0]
        if not row.get('ownedByMe') or row.get('appProperties', {}).get('clv_split_run') != run_hash:
            raise RuntimeError('체크포인트 폴더 소유/실행 범위 불일치')
        parent = row['id']
    else:
        # An ambiguous create is reconciled by the next install's listing.
        parent = api.files().create(body={'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'appProperties': {'clv_split_run': run_hash}}, fields='id').execute()['id']
    original = getattr(screen, '_personal_original_store', screen.SafeProgressStore)
    screen._personal_original_store = original
    run_root = Path(prepared['run_dir']).resolve()

    class PersonalStore(original):
        _personal_storage_installed = True

        def __init__(self, root, ident):
            if (Path(root).resolve() != run_root / 'progress' or ident.seed != 51
                    or ident.config_hash != run_hash
                    or ident.model_id not in {screen.identity(prepared, m).model_id for m in screen.ORDER}):
                raise RuntimeError('체크포인트 저장 범위 밖')
            super().__init__(root, ident)
            self.latest_checkpoint = Path(local_root) / run_hash / self.latest_checkpoint.name
            self.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.remote_epoch = 0

        def checked_epoch(self, path):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            epoch = payload.get('epoch')
            if (payload.get('identity') != asdict(self.identity)
                    or type(epoch) is not int or not 1 <= epoch <= 300
                    or [r.get('epoch') for r in payload.get('history', [])] != list(range(1, epoch+1))
                    or payload.get('completed_weights_only')):
                raise RuntimeError('로컬 checkpoint 신원/epoch/이력 불일치: 파일 보존')
            for key in ('model_state', 'optimizer_state', 'numpy_rng_state',
                        'torch_rng_state', 'cuda_rng_state'):
                if key not in payload:
                    raise RuntimeError('전체 재개 상태 누락: ' + key)
            return epoch

        def publish_epoch(self, epoch):
            current = storage.publish(api, parent, self.latest_checkpoint, self.identity, epoch)
            self.remote_epoch = epoch
            self._write_progress('running', {'epoch': epoch, 'max_epoch': 300,
                'selection': 'none', 'drive_checkpoint_id': current['id'],
                'local_epoch': epoch, 'drive_epoch': epoch, 'drive_interval': drive_interval,
                'checkpoint_sha256': current['appProperties']['clv_sha256'],
                'checkpoint_bytes': int(current['size'])})
            return current

        def restore_epoch(self, model, optimizer, rng):
            candidates = storage.rows_for(api, parent, self.identity)
            local_epoch = self.checked_epoch(self.latest_checkpoint) if self.latest_checkpoint.exists() else 0
            self.remote_epoch = int(candidates[0]['appProperties']['clv_epoch']) if candidates else 0
            if local_epoch and local_epoch >= self.remote_epoch:
                same = [r for r in candidates if int(r['appProperties']['clv_epoch']) == local_epoch]
                if same:
                    digest = hashlib.sha256()
                    with self.latest_checkpoint.open('rb') as source:
                        for chunk in iter(lambda: source.read(8*1024**2), b''):
                            digest.update(chunk)
                    if any(r['appProperties']['clv_sha256'] != digest.hexdigest() for r in same):
                        raise RuntimeError('같은 epoch의 로컬/Drive 내용 불일치: 모두 보존')
                if (not same and (local_epoch % drive_interval == 0 or local_epoch == 300
                        or not self.remote_epoch or local_epoch - self.remote_epoch >= drive_interval)):
                    # Retry the already computed epoch, never overwrite it with an older download.
                    self.publish_epoch(local_epoch)
                print(f'[로컬 재개] epoch{local_epoch}; Drive 확인 epoch{self.remote_epoch}', flush=True)
                return super().restore_epoch(model, optimizer, rng)
            if candidates:
                latest_epoch = int(candidates[0]['appProperties']['clv_epoch'])
                newest = [r for r in candidates if int(r['appProperties']['clv_epoch']) == latest_epoch]
                if len({r['appProperties']['clv_sha256'] for r in newest}) != 1:
                    raise RuntimeError('같은 epoch에 서로 다른 체크포인트 존재: 자동 재개 중단')
                row = newest[0]
                temp = self.latest_checkpoint.with_suffix('.download')
                from googleapiclient.http import MediaIoBaseDownload
                with temp.open('wb') as target:
                    download = MediaIoBaseDownload(target, api.files().get_media(fileId=row['id']),
                                                   chunksize=16*1024**2)
                    done = False
                    while not done:
                        _, done = download.next_chunk(num_retries=3)
                digest = hashlib.sha256()
                with temp.open('rb') as source:
                    for chunk in iter(lambda: source.read(8*1024**2), b''):
                        digest.update(chunk)
                if (temp.stat().st_size != int(row['size']) or
                        digest.hexdigest() != row['appProperties']['clv_sha256']):
                    raise RuntimeError('체크포인트 다운로드 검증 실패: 원본 보존')
                payload = torch.load(temp, map_location='cpu', weights_only=False)
                if payload.get('identity') != asdict(self.identity) or payload.get('epoch') != latest_epoch:
                    raise RuntimeError('체크포인트 실제 신원/epoch 불일치')
                del payload
                self.checked_epoch(temp)
                if self.latest_checkpoint.exists():
                    os.replace(self.latest_checkpoint, self.latest_checkpoint.with_suffix('.previous.pt'))
                temp.replace(self.latest_checkpoint)
            elif self.stage_path.exists() or self.latest_checkpoint.exists():
                raise RuntimeError('진행 이력은 있으나 원격 저장본 없음: 자동 처음부터 학습 금지')
            return super().restore_epoch(model, optimizer, rng)

        def save_epoch(self, model, optimizer, rng, **state):
            start = time.perf_counter()
            epoch = int(state['epoch'])
            previous = self.latest_checkpoint.with_suffix('.previous.pt')
            staged_previous = self.latest_checkpoint.with_suffix('.previous.pending')
            if self.latest_checkpoint.exists():
                if self.checked_epoch(self.latest_checkpoint) >= epoch:
                    raise RuntimeError('로컬 checkpoint epoch 역행: 덮어쓰지 않습니다')
                if staged_previous.exists():
                    raise RuntimeError('미완료 이전본 보존 작업 확인 필요')
                os.link(self.latest_checkpoint, staged_previous)
                os.replace(staged_previous, previous)
            result = super().save_epoch(model, optimizer, rng, **state)
            self.checked_epoch(self.latest_checkpoint)
            local_done = time.perf_counter()
            # A failed boundary upload blocks the next epoch: at most interval-1
            # completed intermediate epochs can be local-only before that boundary.
            if epoch % drive_interval == 0 or epoch == 300 or not self.remote_epoch:
                self.publish_epoch(epoch)
            else:
                self._write_progress('running', {'epoch': epoch, 'max_epoch': 300,
                    'selection': 'none', 'local_epoch': epoch, 'drive_epoch': self.remote_epoch,
                    'drive_interval': drive_interval, 'checkpoint_scope': 'local_only'})
                print(f'[로컬 저장] epoch{epoch}; Drive 저장 확인 epoch{self.remote_epoch}', flush=True)
            self.last_save_timing.update(checkpoint_local_seconds=local_done-start,
                checkpoint_drive_seconds=time.perf_counter()-local_done)
            return result

    screen.SafeProgressStore = PersonalStore
    screen.base.io._atomic_json(run_root/'personal_checkpoint_storage.json', {
        'account': ACCOUNT, 'folder_id': parent, 'folder_name': name,
        'retain_epochs_per_model': 2, 'drive_interval': drive_interval, 'full_optimizer_rng_preserved': True,
        'minimum_free_bytes': MIN_FREE, 'initial_free_bytes': START_FREE,
        'deletion_scope': 'only older verified epoch files created in THIS new run folder; no trash scan',
        'checkpoint_storage_separate_from_reports': True})
    print(f'개인 Drive 여유 {free/1e9:.2f}GB. 로컬 매epoch / Drive {drive_interval}epoch·최종 업로드·검증, 최신+직전 보존. 폴더: {name}', flush=True)
    return parent
