"""Opt-in retention for mounted Drive atomic checkpoint replacements.

Does not change training/source identity. Only deletes explicitly trashed,
owned, same-name/same-parent older copies after live MD5/size verification.
Never empties Trash. One previous copy retained while training, zero on completion.
"""
import hashlib
import json
from pathlib import Path
import time

FIELDS = 'id,name,parents,ownedByMe,trashed,explicitlyTrashed,modifiedTime,size,md5Checksum'


def _quote(s):
    return str(s).replace('\\', '\\\\').replace("'", "\\'")


def _execute(request, *, sleep=time.sleep):
    """Retry quota/transient failures only; do not consume training RNG."""
    for attempt in range(6):
        try:
            return request.execute()
        except Exception as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', None)
            try:
                body = json.loads(getattr(exc, 'content', b'{}'))
                reasons = {e.get('reason') for e in body.get('error', {}).get('errors', [])}
            except (ValueError, TypeError):
                reasons = set()
            limited = status == 429 or (status == 403 and bool(
                reasons & {'rateLimitExceeded', 'userRateLimitExceeded'}))
            if not (limited or status in (500, 502, 503, 504)) or attempt == 5:
                raise
            delay = min(60 * 2**attempt, 480)
            print(f'[Drive 재시도] 일시 제한/서버 오류: {delay}초 대기; 체크포인트 유지', flush=True)
            sleep(delay)


def _list(api, query):
    rows, token = [], None
    while True:
        page = _execute(api.files().list(q=query, spaces='drive', corpora='user', pageSize=1000,
            pageToken=token, fields='nextPageToken,incompleteSearch,files('+FIELDS+')'))
        if page.get('incompleteSearch'):
            raise RuntimeError('체크포인트 목록 불완전: 삭제하지 않습니다')
        rows.extend(page.get('files', []))
        token = page.get('nextPageToken')
        if not token:
            return rows


def prune(api, checkpoint, *, keep_previous=1, retries=30, sleep=time.sleep):
    path = Path(checkpoint)
    if keep_previous not in (0, 1) or not path.name.endswith('_latest.pt'):
        raise ValueError('최신 체크포인트 경로 및 보존수0/1만 허용')
    digest = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            digest.update(block)
    expected, size = digest.hexdigest(), path.stat().st_size
    query = "'me' in owners and name = '"+_quote(path.name)+"'"
    current = None
    for attempt in range(retries):
        rows = _list(api, query + ' and trashed = false')
        if len(rows) == 1:
            f = rows[0]
            if (f.get('ownedByMe') and len(f.get('parents', [])) == 1
                    and f.get('md5Checksum') == expected and int(f.get('size', -1)) == size):
                current = f
                break
        if attempt + 1 < retries:
            sleep(10)
    if current is None:
        raise RuntimeError('최신 체크포인트 Drive 업로드/고유성 확인 실패: 삭제 없이 중단, 이후 재시도하세요')
    parent = current['parents'][0]
    # Parent must still be live; never clean up a user's trashed experiment folder.
    folder = _execute(api.files().get(fileId=parent, fields='id,trashed'))
    if folder.get('trashed') is not False:
        raise RuntimeError('체크포인트 폴더 상태 불명: 삭제하지 않습니다')
    old = _list(api, query + " and trashed = true and '"+_quote(parent)+"' in parents")
    def eligible(f):
        return (f.get('id') != current['id'] and f.get('name') == path.name
            and f.get('ownedByMe') is True and f.get('trashed') is True
            and f.get('explicitlyTrashed') is True and f.get('parents') == [parent]
            and f.get('modifiedTime') and current.get('modifiedTime')
            and f['modifiedTime'] < current['modifiedTime'])
    old = sorted((f for f in old if eligible(f)), key=lambda f:f['modifiedTime'], reverse=True)
    removed = 0
    for candidate in old[keep_previous:]:
        # Recheck both live identity and candidate immediately before permanent deletion.
        live = _execute(api.files().get(fileId=current['id'], fields=FIELDS))
        if (live.get('trashed') is not False or live.get('md5Checksum') != expected
                or live.get('parents') != [parent] or not live.get('ownedByMe')
                or int(live.get('size', -1)) != size):
            raise RuntimeError('최신 체크포인트 동시변경: 추가 삭제를 중단합니다')
        fresh = _execute(api.files().get(fileId=candidate['id'], fields=FIELDS))
        if not eligible(fresh):
            raise RuntimeError('이전 사본 상태 변경: 삭제를 중단합니다')
        api.files().delete(fileId=fresh['id']).execute()
        removed += 1
    if removed:
        print(f'[체크포인트 정리] {path.name}: 이전 휴지통 사본 {removed}개 정리, 최신본 유지', flush=True)
    return removed


def install(screen, api, prepared):
    """Call after prepare, before run, in an idle runtime. Idempotent."""
    import torch
    owner = _execute(api.about().get(fields='user(emailAddress)'))['user']['emailAddress']
    if owner != 'jungunb@u.sogang.ac.kr':
        raise RuntimeError('학교 Drive 계정이 아닙니다')
    module = getattr(screen, 'original', screen)
    cls = module.SafeProgressStore
    if getattr(cls, '_retention_installed', False):
        return
    run_root = Path(prepared['run_dir']).resolve()
    # Validate existing checkpoint payload identities before first cleanup.
    for path in (run_root/'progress'/'resume').glob('*_latest.pt'):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        ident = payload['identity']
        for key in ('config_hash', 'input_hash'):
            if ident[key] != prepared[key]:
                raise RuntimeError('다른 실행 체크포인트: 정리하지 않습니다')
        if ident['source_revision'] != prepared['revision']:
            raise RuntimeError('코드 신원 불일치: 정리하지 않습니다')
        compact = payload.get('completed_weights_only', False)
        if compact and payload.get('epoch') != 300:
            raise RuntimeError('완료 체크포인트 epoch 불일치')
        row_path = run_root/'arms'/f"{ident['model_id']}_s{ident['seed']}.json" if 'model_id' in ident and 'seed' in ident else None
        complete = False
        if row_path is not None and row_path.exists():
            row = json.loads(row_path.read_text())
            complete = row.get('identity') == ident and row.get('epochs') == 300 and payload.get('epoch') == 300
        del payload
        prune(api, path, keep_previous=0 if complete else 1)
    original_save = cls.save_epoch
    def save(self, *args, **kwargs):
        if self.root.resolve() != run_root/'progress':
            raise RuntimeError('보존 정책과 다른 실행 경로')
        # Also clean a previous failed attempt before the quota guard runs.
        if self.latest_checkpoint.exists():
            prune(api, self.latest_checkpoint, keep_previous=1)
        result = original_save(self, *args, **kwargs)
        prune(api, self.latest_checkpoint, keep_previous=1)
        return result
    cls.save_epoch = save
    # mark_complete is also called after final weight compaction, if supported.
    if hasattr(cls, 'mark_complete'):
        original_complete = cls.mark_complete
        def complete(self, *args, **kwargs):
            if self.root.resolve() != run_root/'progress':
                raise RuntimeError('보존 정책과 다른 실행 경로')
            result = original_complete(self, *args, **kwargs)
            if self.latest_checkpoint.exists():
                prune(api, self.latest_checkpoint, keep_previous=0)
            return result
        cls.mark_complete = complete
    cls._retention_installed = True
    print('이 실행의 체크포인트 보존 정책 적용: 최신+직전 유지, 검증된 더 오래된 휴지통 사본만 정리')
