# Existing runtime only: run AFTER interrupting execution, not restarting it.
# Do not re-run input migration/import cells. Then re-run only split.run cell.
import hashlib
import urllib.request
assert 'prepared' in globals() and 'storage' in globals()
assert prepared['config_hash'] == 'ae1edba2afd9'
assert storage.ACCOUNT == 'azsabae@gmail.com'
assert storage.execute(api.about().get(fields='user(emailAddress)'))['user']['emailAddress'] == storage.ACCOUNT
FAST_COMMIT = '5df865e99c449e1369243d79f748d59e4966aa69'
with urllib.request.urlopen(f'https://raw.githubusercontent.com/jung-un/clv-m2-lightgcn-runner/{FAST_COMMIT}/clv_ngcf_storage_fast.py', timeout=60) as r:
    fast_source = r.read()
assert hashlib.sha256(fast_source).hexdigest() == 'd1e03535f7a3b41c9d8617cd095cf0177b119e4968a5eb4c84cfee2ed38b4a6f'
fast_namespace = {}
exec(compile(fast_source, 'clv_ngcf_storage_fast.py', 'exec'), fast_namespace)
fast_namespace['install'](storage)
print('저장 최적화 적용 완료. 기존 학습 셀만 실행하면 마지막 원격 저장 epoch부터 재개합니다.')
