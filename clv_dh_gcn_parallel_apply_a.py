# 기존 단일 학습 셀을 중단한 뒤 같은 런타임에서 실행하세요.
from pathlib import Path
assert 'ctx' in globals() and 'task' in globals(), '기존 준비 셀 필요'
import hashlib, urllib.request, types
url='https://raw.githubusercontent.com/jung-un/clv-m2-lightgcn-runner/982bb7583ce2a46bfa6876e7ba31d60140fd6bee/clv_dh_gcn_parallel.py'
with urllib.request.urlopen(url,timeout=60) as r: payload=r.read()
assert hashlib.sha256(payload).hexdigest()=='e404c7061e52ca1cba2fbba7130d2f0e00eaa67bc8460b152f0cab98e75b9abb'
parallel=types.ModuleType('clv_dh_gcn_parallel')
parallel.__file__='/content/clv_dh_gcn_parallel.py'
Path(parallel.__file__).write_bytes(payload)
exec(compile(payload,parallel.__file__,'exec'),parallel.__dict__)

parallel.install(task,ctx,'A')
status=parallel.run(task,ctx)
print(status)
