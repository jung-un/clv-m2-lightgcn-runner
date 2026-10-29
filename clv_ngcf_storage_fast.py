"""Reduce storage-only requests; retain epoch uploads and existing safety checks.

Install only between training calls. No optimizer, RNG or experiment settings
are changed. Space estimates expire after 120s and deduct every attempted
upload without crediting deletions. A low estimate forces a live check.
"""
import time
import hashlib


def install(storage, ttl=120):
    if getattr(storage, '_fast_storage_installed', False):
        return
    original_check = storage.check_space
    cache = {}

    def check(api, required=0, initial=False):
        key = id(api)
        now = time.monotonic()
        entry = cache.get(key)
        needed = max(storage.START_FREE if initial else storage.MIN_FREE,
                     storage.MIN_FREE + int(required))
        if initial or entry is None or now - entry[0] >= ttl or entry[1] < needed:
            cache.pop(key, None)
            available = original_check(api, required=required, initial=initial)
            entry = [time.monotonic(), available]
            cache[key] = entry
        available = entry[1]
        entry[1] -= int(required)  # conservative even when upload fails
        return available

    def publish(api, parent, path, ident, epoch):
        try:
            from googleapiclient.http import MediaFileUpload
            size = path.stat().st_size
            sha, md5 = hashlib.sha256(), hashlib.md5()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024**2), b''):
                    sha.update(block)
                    md5.update(block)
            props = dict(clv_split_run=ident.config_hash, clv_model=ident.model_id,
                         clv_seed=str(ident.seed), clv_epoch=str(epoch), clv_sha256=sha.hexdigest())
            rows = storage.rows_for(api, parent, ident)  # always fresh
            same = [r for r in rows if int(r['appProperties']['clv_epoch']) == epoch]
            if same and any(r['appProperties']['clv_sha256'] != sha.hexdigest() for r in same):
                raise RuntimeError('동일 epoch의 다른 checkpoint: 덮어쓰기 금지')
            if same:
                current = same[0]
            else:
                storage.wait_for_space(api, size)
                # Ambiguous creates are NOT retried; next call lists afresh.
                current = api.files().create(body={
                    'name': f'{ident.model_id}_s{ident.seed}_ep{epoch:03d}.pt',
                    'parents': [parent], 'appProperties': props},
                    media_body=MediaFileUpload(str(path), mimetype='application/octet-stream',
                                               resumable=True), fields=storage.FIELDS).execute()
            live = storage.execute(api.files().get(fileId=current['id'], fields=storage.FIELDS))
            for row in (current, live):
                if (row.get('ownedByMe') is not True or row.get('trashed') is not False
                        or row.get('parents') != [parent] or row.get('appProperties') != props
                        or int(row.get('size', -1)) != size or row.get('md5Checksum') != md5.hexdigest()):
                    raise RuntimeError('새 checkpoint 저장 검증 실패: 이전본 보존')
            # Replace redundant second folder listing with the independently
            # verified upload. Unseen concurrent files are never deleted.
            rows = [r for r in rows if r['id'] != live['id']] + [live]
            keep = set(sorted({int(r['appProperties']['clv_epoch']) for r in rows}, reverse=True)[:2])
            for row in rows:
                if int(row['appProperties']['clv_epoch']) in keep:
                    continue
                fresh = storage.execute(api.files().get(fileId=row['id'], fields=storage.FIELDS))
                if any(fresh.get(k) != row.get(k) for k in
                       ('parents', 'appProperties', 'md5Checksum', 'size', 'trashed', 'ownedByMe')):
                    raise RuntimeError('이전 checkpoint 동시변경: 삭제 중단')
                storage.execute(api.files().delete(fileId=row['id']))
            return live
        except BaseException:
            cache.pop(id(api), None)
            raise

    storage.check_space = check
    storage.publish = publish
    storage._fast_storage_installed = True
    print(f'[저장 최적화] 용량 조회 {ttl}초 캐시·업로드 용량 차감. '
          '매epoch 업로드·해시 검증·최신/직전 보존·재시도 유지.', flush=True)
