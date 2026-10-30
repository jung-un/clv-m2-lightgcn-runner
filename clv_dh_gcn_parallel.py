"""Disjoint scheduling only; frozen task identities and training remain unchanged."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ASSIGNMENTS = {'A': (51, 42, 43, 44, 45), 'B': (46, 47, 48)}


def assignment(task, ctx):
    flat = [s for seeds in ASSIGNMENTS.values() for s in seeds]
    if len(flat) != len(set(flat)) or set(flat) != set(task.SEEDS):
        raise RuntimeError('시드 중복 또는 누락')
    return {'version': 1, 'assignments': {k:list(v) for k,v in ASSIGNMENTS.items()},
            'protocol_sha256': task.sha(ctx['out']/'protocol.json'),
            'scheduler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def install(task, ctx, role):
    if role not in ASSIGNMENTS:
        raise ValueError('role must be A or B')
    expected = assignment(task, ctx)
    marker = ctx['out']/'parallel_assignment.json'
    # Called in an idle notebook cell, after the previous unsplit training cell stops.
    # B fails closed until A has installed this exact partition.
    if role == 'B' and not marker.exists():
        raise RuntimeError('기존 GPU A 분할 적용 확인 전: B 학습 차단')
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise RuntimeError('기존 시드 배정 불일치; 파일 보존')
    if role == 'A' and not marker.exists():
        ctx['original'].base.io._atomic_json(marker, expected)
    ctx['parallel_role'] = role
    ctx['parallel_assignment'] = expected
    print(f'[시드 고정] GPU {role}: {ASSIGNMENTS[role]}; 완료49·50 재사용', flush=True)


def check(ctx):
    if json.loads((ctx['out']/'parallel_assignment.json').read_text()) != ctx['parallel_assignment']:
        raise RuntimeError('시드 배정 변경 감지; 기존 결과 보존')


def report(task, ctx):
    import pandas as pd
    o,p = ctx['original'],ctx['shared']
    rows = list(ctx['previous'])
    rows += [r for label in task.ORDER if (r := o.completed(p,label)) is not None]
    for path in sorted((ctx['out']/'arms').glob('*.json')):
        row=json.loads(path.read_text())
        if row['seed'] not in range(42,49) or row['model'] not in task.ORDER:
            raise RuntimeError('범위 밖 결과')
        task.validate_row(row,row['seed'],row['model'],p['references']['metrics'])
        rows.append(row)
    keys={(r['seed'],r['model']) for r in rows}
    if len(keys)!=len(rows):raise RuntimeError('중복 결과')
    missing=[[s,m] for s in task.ALL_SEEDS for m in task.ORDER if (s,m) not in keys]
    role=ctx['parallel_role']
    status={'role':role,'assigned_seeds':list(ASSIGNMENTS[role]),'completed':len(rows),
            'planned':80,'complete':not missing,'missing':missing,'fixed_epoch':300,
            'worker_complete':not any(s in ASSIGNMENTS[role] for s,m in missing),
            'significance_claim':False}
    out=ctx['out']/'parallel_reports'/role
    out.mkdir(parents=True,exist_ok=True)
    absolute=pd.DataFrame([{'seed':r['seed'],'model':r['model'],**r['metrics']} for r in rows])
    o.base.io._atomic_csv(out/'absolute.csv',absolute)
    o.base.io._atomic_json(out/'status.json',status)
    if not missing:
        summary=absolute.groupby('model')[list(p['references']['metrics'])].agg(['mean','std'])
        summary.columns=[f'{m}__{stat}' for m,stat in summary.columns]
        o.base.io._atomic_csv(out/'mean_std_10seeds.csv',summary.reindex(task.ORDER).reset_index())
    return status,rows


def run(task, ctx):
    # Training is serial inside each runtime; this lock rejects duplicate threads locally.
    import threading
    lock=ctx.setdefault('_parallel_lock',threading.Lock())
    if not lock.acquire(blocking=False):raise RuntimeError('동일 런타임 중복 실행')
    try:
        for seed in ASSIGNMENTS[ctx['parallel_role']]:
            for label in task.ORDER:
                check(ctx)
                print(f"[학교 GCN GPU {ctx['parallel_role']}] seed{seed} {label}",flush=True)
                if seed==51:ctx['original'].run_arm(ctx['shared'],label)
                else:task.run_new(ctx,seed,label)
                status,_=report(task,ctx)
                print(f"[전체] {status['completed']}/80 완료",flush=True)
        return report(task,ctx)[0]
    finally:lock.release()


def export(task,ctx):
    status,rows=report(task,ctx)
    out=ctx['out']/'parallel_reports'/ctx['parallel_role']
    path=out/('dh_gcn_final10_results'+('' if status['complete'] else '_partial')+'.zip')
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        z.write(ctx['out']/'protocol.json','protocol.json')
        z.write(ctx['out']/'parallel_assignment.json','parallel_assignment.json')
        for name in ('status.json','absolute.csv','mean_std_10seeds.csv'):
            f=out/name
            if f.exists() and (name!='mean_std_10seeds.csv' or status['complete']):z.write(f,name)
        for row in rows:z.writestr(f"arms/{row['model_id']}_s{row['seed']}.json",json.dumps(row,ensure_ascii=False))
    return str(path)
