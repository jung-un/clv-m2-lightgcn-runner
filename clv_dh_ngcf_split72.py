"""Operational 7+2 dispatch; the frozen NGCF runner/model/loss remain untouched."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

SOURCE_COMMIT = '63146eff4ed60bfb0cc430b77d58fdc40e4db7c5'
ASSIGNMENTS = {'personal': tuple(range(42, 49)), 'school': (50, 51)}
VERSION = 'ngcf-split-7-2-v1'
SCIENCE_KEYS = ('code_version', 'source_revision', 'code_sha256', 'input_hash',
    'data_stats', 'split', 'feature_sha256', 'm3_audit', 'm4_weight_audit',
    'dataset_initialization', 'fixed_intervention_settings', 'models', 'backbone',
    'propagation', 'adjacency', 'regularization', 'm2_adaptation', 'initialization',
    'fixed_epoch', 'validation', 'holdout', 'early_stopping')
PATH_KEYS = ('out_dir', 'm2_reference', 'm4_reference', 'seed49_reference')


def core():
    return importlib.import_module('clv_dh_ngcf_remaining9')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scientific_signature(protocol):
    cfg = {k: v for k, v in protocol['config'].items() if k not in PATH_KEYS}
    if cfg['seeds'] != list(range(42, 52)) or cfg['epochs'] != 300:
        raise RuntimeError('최종10시드/300epoch 계획 변경')
    if protocol['source_revision'] != SOURCE_COMMIT:
        raise RuntimeError('고정 NGCF 학습 source 변경')
    if any(protocol[k] is not False for k in ('validation', 'holdout', 'early_stopping')):
        raise RuntimeError('검증/holdout/조기종료 변경')
    return {**{k: protocol[k] for k in SCIENCE_KEYS}, 'config': cfg,
            'seed49_sha256': protocol['seed49_audit']['sha256']}


def receipt(prepared, role):
    if role not in ASSIGNMENTS:
        raise ValueError('계정 역할은 personal 또는 school만 허용')
    protocol = prepared['protocol']
    if digest(protocol)[:12] != prepared['config_hash']:
        raise RuntimeError('원래 protocol/config_hash 불일치')
    return {'version': VERSION, 'role': role, 'seeds': list(ASSIGNMENTS[role]),
            'config_hash': prepared['config_hash'],
            'science_sha256': digest(scientific_signature(protocol)),
            'source_revision': prepared['revision'], 'input_hash': prepared['input_hash']}


def install(prepared, role):
    """Guard run_arm itself, so even an old screen.run cannot train foreign seeds."""
    s = core()
    s.validate_config(prepared['cfg'])
    rec = receipt(prepared, role)
    root = Path(prepared['run_dir'])
    sidecar = root / 'split_assignment.json'
    if sidecar.exists() and json.loads(sidecar.read_text()) != rec:
        raise RuntimeError('이 출력의 기존 시드 배정이 다릅니다. 변경하지 않습니다')
    foreign = set(s.NEW_SEEDS) - set(ASSIGNMENTS[role])
    for seed in foreign:
        if any((root / 'arms').glob(f'*_s{seed}.json')) or any(
                (root / 'progress' / 'stages').glob(f'*_s{seed}.*')):
            raise RuntimeError(f'다른 담당 seed{seed}의 기존 작업 발견: 먼저 중복 여부 확인')
    s.base.io._atomic_json(sidecar, rec)
    original = getattr(s, '_split_original_run_arm', s.run_arm)
    s._split_original_run_arm = original

    def guarded(p, label, seed):
        if Path(p['run_dir']) != root or receipt(p, role) != rec:
            raise RuntimeError('시드 배정과 다른 실행/입력')
        if seed not in ASSIGNMENTS[role]:
            raise RuntimeError(f'{role}은 seed{seed} 학습 금지; 담당={ASSIGNMENTS[role]}')
        return original(p, label, seed)

    s.run_arm = guarded
    s.run = lambda p: run(p, role)
    print(f'[시드 배정 고정] {role}: {ASSIGNMENTS[role]}; seed49 재사용', flush=True)
    return rec


def check_personal_receipt(prepared, path):
    rec = json.loads(Path(path).read_text())
    expected = receipt(prepared, 'school')
    if (rec.get('version') != VERSION or rec.get('role') != 'personal'
            or rec.get('seeds') != list(ASSIGNMENTS['personal'])
            or rec.get('science_sha256') != expected['science_sha256']
            or rec.get('source_revision') != SOURCE_COMMIT):
        raise RuntimeError('개인7시드 배정 확인서/학습조건 불일치: 학교 실행 금지')
    return rec


def status(prepared, role):
    s = core()
    rows = [r for seed in ASSIGNMENTS[role] for label in s.ORDER
            if (r := s.completed(prepared, label, seed)) is not None]
    return {'role': role, 'seeds': list(ASSIGNMENTS[role]), 'complete':
            len(rows) == 8 * len(ASSIGNMENTS[role]), 'completed': len(rows),
            'planned': 8 * len(ASSIGNMENTS[role]), 'global_ten_seed_complete': False}


def run(prepared, role):
    install(prepared, role)
    s = core()
    for seed in ASSIGNMENTS[role]:
        for label in s.ORDER:
            print(f'[{role}] NGCF seed{seed} {label}', flush=True)
            s.run_arm(prepared, label, seed)
            s.report(prepared)
            print(json.dumps(status(prepared, role), ensure_ascii=False), flush=True)
    return status(prepared, role)


def export_partition(prepared, role, destination):
    """Result-only portable packet; full checkpoints stay in their owning account."""
    s = core()
    rec = receipt(prepared, role)
    saved = json.loads((Path(prepared['run_dir']) / 'split_assignment.json').read_text())
    if saved != rec:
        raise RuntimeError('시드 배정 미적용: 내보내기 금지')
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
        z.writestr('protocol.json', json.dumps(prepared['protocol']))
        z.writestr('split_assignment.json', json.dumps(rec))
        for seed in ASSIGNMENTS[role]:
            for label in s.ORDER:
                if s.completed(prepared, label, seed) is not None:
                    f = s.arm_path(prepared, label, seed)
                    z.write(f, 'arms/' + f.name)
        for name, blob in prepared['previous']['blobs'].items():
            z.writestr('reused/seed49/' + name, blob)
    temporary.replace(path)
    return str(path)


def _packet(path, role):
    s = core()
    with ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any('..' in Path(n).parts for n in names):
            raise RuntimeError('중복/잘못된 ZIP 경로')
        protocol = json.loads(z.read('protocol.json'))
        rec = json.loads(z.read('split_assignment.json'))
        signature = scientific_signature(protocol)
        chash = digest(protocol)[:12]
        if (rec.get('role') != role or rec.get('seeds') != list(ASSIGNMENTS[role])
                or rec.get('config_hash') != chash or rec.get('version') != VERSION
                or rec.get('science_sha256') != digest(signature)):
            raise RuntimeError('배정/원본 protocol 지문 불일치')
        seed49 = {}
        for name, sha in s.OLD_SHA.items():
            b = z.read('reused/seed49/' + name)
            if hashlib.sha256(b).hexdigest() != sha:
                raise RuntimeError('seed49 원본 변경')
            seed49[name] = b
        metrics = list(json.loads(seed49['arms/ngcf_m1_lastweek_s49.json'])['metrics'])
        rows = []
        for name in names:
            if not name.startswith('arms/'):
                continue
            row = json.loads(z.read(name))
            seed, label = row['seed'], row['model']
            if seed not in ASSIGNMENTS[role] or label not in s.ORDER:
                raise RuntimeError('다른 계정 담당 시드가 섞였습니다')
            ident = {'stage': s.VERSION, 'model_id': f'ngcf_{label.lower()}_lastweek',
                     'seed': seed, 'config_hash': chash,
                     'source_revision': protocol['source_revision'],
                     'input_hash': protocol['input_hash']}
            s._validate_row(row, label, seed, metrics, ident)
            if row['diagnostics']['feature_sha256'] != protocol['feature_sha256']:
                raise RuntimeError('특징 지문 불일치')
            if name != f'arms/ngcf_{label.lower()}_lastweek_s{seed}.json':
                raise RuntimeError('결과 파일 이름/신원 불일치')
            rows.append(row)
    if len(rows) != len({(r['model'], r['seed']) for r in rows}):
        raise RuntimeError('모델/시드 중복')
    return protocol, signature, rows, seed49, metrics


def merge(personal_zip, school_zip, destination):
    """Preserve original identities; never rewrite school rows as personal fits."""
    import pandas as pd
    s = core()
    a, b = _packet(personal_zip, 'personal'), _packet(school_zip, 'school')
    if a[1] != b[1] or a[4] != b[4] or a[3] != b[3]:
        raise RuntimeError('두 계정 학습조건/입력/seed49 원본 불일치')
    rows = a[2] + b[2] + [json.loads(a[3][f'arms/ngcf_{m.lower()}_lastweek_s49.json'])
                              for m in s.ORDER]
    pairs = [(r['model'], r['seed']) for r in rows]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError('모델/시드 중복: 합산하지 않습니다')
    absolute = pd.DataFrame([{'model': r['model'], 'model_id': r['model_id'],
        'seed': r['seed'], 'source': ('reused_seed49' if r['seed'] == 49 else
        'personal7' if r['seed'] in ASSIGNMENTS['personal'] else 'school2'),
        **r['metrics']} for r in rows]).sort_values(['seed', 'model'])
    tables = {'absolute': absolute, 'diagnostics': pd.DataFrame([
        {'model': r['model'], 'seed': r['seed'], **r['diagnostics']} for r in rows])}
    tables.update(s.aggregate(absolute, a[4]))
    missing = [f'seed{seed}:{m}' for seed in s.SEEDS for m in s.ORDER if (m, seed) not in pairs]
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        s.base.io._atomic_csv(root / f'{name}.csv', table)
    # Retain source protocols and row identities, rather than fabricating one run ID.
    s.base.io._atomic_json(root / 'source_protocols.json', {'personal': a[0], 'school': b[0]})
    for row in rows:
        s.base.io._atomic_json(root / 'arms' / f"{row['model_id']}_s{row['seed']}.json", row)
    reading = {'complete': not missing, 'completed': len(rows), 'planned': 80,
               'missing': missing, 'mean_std_available': not missing, 'sample_std_ddof': 1,
               'significance_claim': False, 'retuning_allowed': False,
               'note': 'Fixed split by seed, not by observed performance. Original identities preserved.'}
    s.base.io._atomic_json(root / 'result.json', reading)
    return {'reading': reading, **tables}
