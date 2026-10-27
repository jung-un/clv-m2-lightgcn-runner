"""School-only bounded checkpoint publication; no model/optimizer math changes.

Each saved epoch is an immutable Drive file. Only this new run's verified files
older than its latest TWO epochs are removed. Existing experiments and Trash
are never scanned for deletion. Full Adam/RNG is retained, including epoch300.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

ACCOUNT = 'jungunb@u.sogang.ac.kr'
MIN_FREE = 1_000_000_000
START_FREE = 1_000_000_000  # User-approved reserve; each upload also budgets its size.
FIELDS = 'id,name,size,md5Checksum,parents,ownedByMe,trashed,appProperties'
LOCAL_CACHE = Path('/content/ngcf_school_local')


class InsufficientStorage(RuntimeError):
    def __init__(self, available, needed):
        self.available, self.needed = available, needed
        super().__init__(f'학교 Drive 여유 {available/1e9:.2f}GB < 필요 {needed/1e9:.2f}GB. 기존 파일을 보존하고 중단합니다.')


def execute(request):
    # No training RNG is consumed by backoff. Mutating create/upload requests
    # deliberately do not use this helper (ambiguous success must be reconciled).
    for attempt in range(9):
        try:
            return request.execute()
        except Exception as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', None)
            try:
                details = json.loads(getattr(exc, 'content', b'{}'))
                reasons = {e.get('reason') for e in details.get('error', {}).get('errors', [])}
            except (ValueError, TypeError, AttributeError):
                reasons = set()
            limited = status == 429 or (status == 403 and bool(
                reasons & {'rateLimitExceeded', 'userRateLimitExceeded'}))
            if not (limited or status in (500, 502, 503, 504)) or attempt == 8:
                raise
            delay = min(60, (15 if limited else 2) * 2 ** attempt)
            print(f'[Drive 재시도] HTTP {status}: {delay}초 대기 ({attempt + 1}/8). 저장 확인 전 학습 진행 안 함.', flush=True)
            time.sleep(delay)


def listing(api, query, fields=FIELDS):
    rows, token = [], None
    while True:
        page = execute(api.files().list(q=query, spaces='drive', corpora='user',
            pageSize=1000, pageToken=token,
            fields='nextPageToken,incompleteSearch,files(' + fields + ')'))
        if page.get('incompleteSearch'):
            raise RuntimeError('Drive 목록 불완전: 저장/정리 중단')
        rows.extend(page.get('files', []))
        token = page.get('nextPageToken')
        if not token:
            return rows


def check_space(api, required=0, initial=False):
    info = execute(api.about().get(fields='user(emailAddress),storageQuota'))
    if info['user']['emailAddress'] != ACCOUNT:
        raise RuntimeError('학교 Drive 계정으로 인증해야 합니다')
    rows = listing(api, "'me' in owners", 'id,mimeType,quotaBytesUsed')
    if len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError('Drive 용량 목록 중복')
    used = 0
    for r in rows:
        if r['mimeType'] in ('application/vnd.google-apps.folder', 'application/vnd.google-apps.shortcut'):
            continue
        if 'quotaBytesUsed' not in r or int(r['quotaBytesUsed']) < 0:
            raise RuntimeError('파일별 용량 확인 실패')
        used += int(r['quotaBytesUsed'])
    available = 10_000_000_000 - used
    quota = info.get('storageQuota', {})
    if 'limit' in quota and 'usage' in quota:
        available = min(available, int(quota['limit']) - int(quota['usage']))
    needed = max(START_FREE if initial else MIN_FREE, MIN_FREE + int(required))
    if available < needed:
        raise InsufficientStorage(available, needed)
    return available


def wait_for_space(api, required=0, initial=False, timeout=600, interval=30):
    """Keep the reserve; wait for another experiment's temporary save to finish."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check_space(api, required=required, initial=initial)
        except InsufficientStorage as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            delay = min(interval, remaining)
            print(f'[Drive 공간 대기] 여유 {exc.available/1e9:.2f}GB / 필요 {exc.needed/1e9:.2f}GB. '
                  f'기준 유지·다음 학습/업로드 보류; {delay:.0f}초 후 실제 용량 재검사.', flush=True)
            time.sleep(delay)


def folder(api, run_hash):
    name = 'ngcf_school2_checkpoints_' + run_hash
    query = ("'me' in owners and trashed = false and name = '" + name +
             "' and mimeType = 'application/vnd.google-apps.folder'")
    rows = listing(api, query)
    if len(rows) > 1:
        raise RuntimeError('동일 학교 checkpoint 폴더 중복')
    if rows:
        row = rows[0]
        if row.get('appProperties', {}).get('clv_split_run') != run_hash:
            raise RuntimeError('기존 폴더 소유 범위 불일치')
        return row['id']
    return execute(api.files().create(body={'name': name, 'mimeType':
        'application/vnd.google-apps.folder', 'appProperties': {'clv_split_run': run_hash}},
        fields='id'))['id']


def rows_for(api, parent, ident):
    rows = listing(api, "'" + parent + "' in parents and trashed = false")
    result = []
    for row in rows:
        p = row.get('appProperties', {})
        if p.get('clv_model') != ident.model_id or p.get('clv_seed') != str(ident.seed):
            continue
        if (row.get('ownedByMe') is not True or row.get('parents') != [parent]
                or p.get('clv_split_run') != ident.config_hash
                or not p.get('clv_sha256') or not row.get('md5Checksum')
                or not 1 <= int(p.get('clv_epoch', 0)) <= 300):
            raise RuntimeError('학교 checkpoint 메타데이터 불일치')
        result.append(row)
    return sorted(result, key=lambda r: int(r['appProperties']['clv_epoch']), reverse=True)


def publish(api, parent, path, ident, epoch):
    from googleapiclient.http import MediaFileUpload
    size = path.stat().st_size
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            sha.update(b); md5.update(b)
    props = {'clv_split_run': ident.config_hash, 'clv_model': ident.model_id,
             'clv_seed': str(ident.seed), 'clv_epoch': str(epoch), 'clv_sha256': sha.hexdigest()}
    rows = rows_for(api, parent, ident)
    same = [r for r in rows if int(r['appProperties']['clv_epoch']) == epoch]
    if same and any(r['appProperties']['clv_sha256'] != sha.hexdigest() for r in same):
        raise RuntimeError('동일 epoch의 다른 checkpoint 발견: 덮어쓰지 않습니다')
    if same:
        current = same[0]
    else:
        wait_for_space(api, size)
        request = api.files().create(body={'name': f'{ident.model_id}_s{ident.seed}_ep{epoch:03d}.pt',
            'parents': [parent], 'appProperties': props}, media_body=MediaFileUpload(
            str(path), mimetype='application/octet-stream', resumable=True), fields=FIELDS)
        # Do not blindly retry files.create: an ambiguous success is recovered by listing next run.
        current = request.execute()
    if int(current.get('size', -1)) != size or current.get('md5Checksum') != md5.hexdigest():
        raise RuntimeError('업로드 크기/MD5 불일치: 이전 checkpoint 보존')
    live = execute(api.files().get(fileId=current['id'], fields=FIELDS))
    if (live.get('trashed') is not False or live.get('appProperties') != props
            or live.get('md5Checksum') != md5.hexdigest()):
        raise RuntimeError('새 checkpoint 영구 저장 확인 실패')
    rows = rows_for(api, parent, ident)
    epochs = sorted({int(r['appProperties']['clv_epoch']) for r in rows}, reverse=True)
    keep = set(epochs[:2])
    for row in rows:
        if int(row['appProperties']['clv_epoch']) in keep:
            continue
        fresh = execute(api.files().get(fileId=row['id'], fields=FIELDS))
        if any(fresh.get(k) != row.get(k) for k in ('parents', 'appProperties', 'md5Checksum', 'size', 'trashed')):
            raise RuntimeError('이전 checkpoint 동시변경: 삭제하지 않습니다')
        execute(api.files().delete(fileId=row['id']))
    return current


def install(prepared, api):
    import clv_dh_ngcf_remaining9 as s
    from googleapiclient.http import MediaIoBaseDownload
    import torch
    wait_for_space(api, initial=True)
    parent = folder(api, prepared['config_hash'])
    original = getattr(s.original, '_school_original_store', s.original.SafeProgressStore)
    s.original._school_original_store = original
    run_root = Path(prepared['run_dir'])

    class SchoolStore(original):
        def __init__(self, root, ident):
            if Path(root) != run_root / 'progress' or ident.seed not in (50, 51):
                raise RuntimeError('학교 checkpoint 저장 범위 밖')
            super().__init__(root, ident)
            self.latest_checkpoint = LOCAL_CACHE / ident.config_hash / self.latest_checkpoint.name
            self.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        def restore_epoch(self, model, optimizer, rng):
            rows = rows_for(api, parent, self.identity)
            if rows:
                row = rows[0]
                temp = self.latest_checkpoint.with_suffix('.download')
                with temp.open('wb') as target:
                    download = MediaIoBaseDownload(target, api.files().get_media(fileId=row['id']))
                    done = False
                    while not done:
                        _, done = download.next_chunk(num_retries=3)
                if (temp.stat().st_size != int(row['size']) or
                        hashlib.sha256(temp.read_bytes()).hexdigest() != row['appProperties']['clv_sha256']):
                    raise RuntimeError('학교 checkpoint 다운로드 검증 실패')
                payload = torch.load(temp, map_location='cpu', weights_only=False)
                if (payload.get('identity') != asdict(self.identity) or
                        payload.get('epoch') != int(row['appProperties']['clv_epoch'])):
                    raise RuntimeError('학교 checkpoint 실제 신원 불일치')
                temp.replace(self.latest_checkpoint)
            elif self.stage_path.exists():
                raise RuntimeError('진행 이력은 있으나 학교 영구 checkpoint가 없습니다')
            else:
                # Never use a stale un-published local file as a durable checkpoint.
                if self.latest_checkpoint.exists():
                    raise RuntimeError('원격 저장이 없는 로컬 checkpoint 발견: 수동 확인 필요')
            return super().restore_epoch(model, optimizer, rng)

        def save_epoch(self, model, optimizer, rng, **state):
            # Local serialization does not consume Drive space. publish() checks
            # fresh account usage once, against the actual upload size.
            result = super().save_epoch(model, optimizer, rng, **state)
            current = publish(api, parent, self.latest_checkpoint, self.identity, int(state['epoch']))
            self._write_progress('running', {'epoch': state['epoch'], 'max_epoch': 300,
                'selection': 'none', 'drive_checkpoint_id': current['id'],
                'checkpoint_sha256': current['appProperties']['clv_sha256']})
            return result

    s.original.SafeProgressStore = SchoolStore
    s.base.io._atomic_json(run_root / 'school_checkpoint_storage.json', {
        'account': ACCOUNT, 'folder_id': parent, 'retain_epochs_per_model': 2,
        'full_optimizer_rng_preserved': True, 'initial_free_bytes': START_FREE,
        'minimum_free_bytes': MIN_FREE, 'uses_drive_api_immutable_epoch_files': True})
    print('학교 저장: 매epoch 영구 업로드/해시검증, 모형별 최신+직전 전체 상태 보존', flush=True)
    return parent
