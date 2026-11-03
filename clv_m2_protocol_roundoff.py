"""Allow diagnostic reduction roundoff only; preserve frozen protocol bytes."""
import ast
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path

LIMITS = {'item_v_std': 1e-12,
          'user_n_reliability_mean': 2.5e-7,
          'user_v_reliability_mean': 2.5e-7}


def differences(saved, current):
    old, new = json.loads(json.dumps(saved)), json.loads(json.dumps(current))
    a, b = old.pop('feature_diagnostics'), new.pop('feature_diagnostics')
    if old != new or set(a) != set(b):
        raise RuntimeError('고정 protocol의 입력/설정/특징/진단 항목이 다릅니다')
    result = {}
    for key in a:
        if a[key] == b[key]:
            continue
        x, y = a[key], b[key]
        if (key not in LIMITS or type(x) not in (int, float)
                or type(y) not in (int, float) or not math.isfinite(x)
                or not math.isfinite(y) or abs(x-y) > LIMITS[key]):
            raise RuntimeError('허용하지 않은 feature 진단 차이: ' + key)
        result[key] = {'saved': x, 'current': y, 'atol': LIMITS[key]}
    return result


def guard(path, current, atomic_json):
    path = Path(path)
    if not path.exists():
        atomic_json(path, current)
        return current
    original = path.read_bytes()
    saved = json.loads(original)
    delta = differences(saved, current)
    if delta:
        digest = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
        audit = path.with_name('diagnostic_roundoff_' + digest[:16] + '.json')
        atomic_json(audit, {'protocol_sha256': hashlib.sha256(original).hexdigest(),
                           'differences': delta, 'current_feature_diagnostics': current['feature_diagnostics'],
                           'unchanged': 'all other protocol fields including exact feature/input hashes and initialization'})
        print('[protocol 확인] 입력·특징·설정 동일; 진단 소수점 차이만 기록, 기존 원본 보존', flush=True)
    assert path.read_bytes() == original
    return saved


OLD = '''    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("기존 실행 경로의 고정 protocol이 다릅니다")
    io._atomic_json(protocol_path, protocol)'''
NEW = '''    protocol = _protocol_roundoff_guard(protocol_path, protocol, io._atomic_json)'''


def patched_source(module):
    source = inspect.getsource(getattr(module, '_before_roundoff_prepare', module.prepare))
    if source.count(OLD) != 1:
        raise RuntimeError('고정 prepare 소스가 예상과 다릅니다; 수정하지 않습니다')
    return source.replace(OLD, NEW)


def install(module):
    if getattr(module.prepare, '_roundoff_guard_installed', False):
        return
    scope = dict(module.__dict__, _protocol_roundoff_guard=guard)
    source = patched_source(module)
    exec(compile(source, '<m2-protocol-roundoff>', 'exec'), scope)
    module._before_roundoff_prepare = module.prepare
    module.prepare = scope['prepare']
    module.prepare._roundoff_guard_installed = True


def recover_frame(module, frame):
    """Finish only the already-computed input stage; no graph reallocation."""
    if (frame.f_code.co_name != 'prepare'
            or Path(frame.f_code.co_filename).resolve() != Path(module.__file__).resolve()):
        raise RuntimeError('입력 준비 오류 프레임 신원 불일치')
    values = dict(frame.f_locals)
    module._validate_config(values['cfg'])
    assert module._feature_hash(values['features']) == values['feature_sha']
    differences(json.loads(values['protocol_path'].read_text()), values['protocol'])
    function = ast.parse(patched_source(module)).body[0]
    start = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'protocol_path' for t in node.targets))
    body = function.body[start:]
    assert isinstance(body[-1], ast.Return)
    body[-1] = ast.Assign(targets=[ast.Name(id='_recovered', ctx=ast.Store())], value=body[-1].value)
    scope = dict(module.__dict__, **values, _protocol_roundoff_guard=guard)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 '<finish-verified-m2-inputs>', 'exec'), scope)
    recovered = scope['_recovered']
    assert recovered['data'] is values['data']
    return recovered, asdict(values['cfg'])
