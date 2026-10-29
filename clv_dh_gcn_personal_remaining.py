"""Personal-Drive coordinator; numerical training stays in the frozen seed51 code.

Run in a checkout of 05237af3c3c1. Restore seed51 under its original relative
MyDrive path (the mounted ACCOUNT changes, not the scientific run identity).
Never initialize a different seed with seed51 weights.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

SOURCE = '05237af3c3c13883f6e4d59cce1d1b641af5caed'
SEEDS = (51, 42, 43, 44, 45, 46, 47, 48)
ALL_SEEDS = tuple(range(42, 52))
ORDER = ('M1', 'M2', 'M3', 'M4', 'M5_A', 'M5_B', 'M5_C', 'M5_D')
VERSION = 'clv-dh-gcn-personal-remaining-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def unpack_verified(bundle, destination):
    """Manifest-verified inputs, no replacement of a different existing file."""
    destination = Path(destination)
    with ZipFile(bundle) as z:
        manifest = json.loads(z.read('manifest.json'))
        for name, digest in manifest.items():
            p = Path(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe archive path')
            target = destination / p
            if target.exists():
                if sha(target) != digest:
                    raise RuntimeError(f'기존 파일 불일치; 보존하고 중단: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.importing')
            with z.open(name) as source, temp.open('wb') as sink:
                shutil.copyfileobj(source, sink)
            if sha(temp) != digest:
                raise RuntimeError(f'압축 파일 해시 불일치: {name}')
            temp.replace(target)
    return manifest


def validate_row(row, seed, label, metrics=None):
    import numpy as np
    if (row.get('seed') != seed or row.get('model') != label
            or row.get('model_id') != f'gcn_{label.lower()}_lastweek'
            or row.get('epochs') != 300
            or [r['epoch'] for r in row.get('training_history', [])] != list(range(1, 301))
            or len(row.get('metrics', {})) != 142
            or not np.isfinite(list(row['metrics'].values())).all()):
        raise RuntimeError(f'완료 결과 불일치: seed{seed} {label}')
    if metrics is not None and set(metrics) != set(row['metrics']):
        raise RuntimeError('평가 지표 목록 불일치')


def prepare(assets, out_dir):
    import clv_dh_gcn_final8_seed51 as original
    assets, out_dir = Path(assets), Path(out_dir)
    original_protocol = json.loads((assets/'seed51/protocol.json').read_text())
    for name, digest in original_protocol['code_sha256'].items():
        if sha(Path(original.__file__).with_name(name)) != digest:
            raise RuntimeError(f'고정 학습 코드 불일치: {name}')
    local = Path('/content/dh_gcn_school_inputs_v1')
    unpack_verified(assets/'dh_gcn_school_inputs_v1.zip', local)
    original.base.v3.SCHEMA['dunnhumby'].update(
        tx_path=str(local/'transaction_data.csv'), item_meta_path=str(local/'product.csv'))
    config = dict(original_protocol['config'])
    config['seeds'] = tuple(config['seeds'])
    cfg = original.Config(**config)
    # Personal MyDrive has the historical RELATIVE path solely to retain identity.
    run51 = Path(cfg.out_dir)/'3a93e11f5f4b'
    marker = run51/'personal_migration.json'
    if not marker.exists():
        if run51.exists():
            raise RuntimeError('기존 seed51 폴더 발견: 임의 덮어쓰기 없이 확인 필요')
        shutil.copytree(assets/'seed51', run51)
        original.base.io._atomic_json(marker, {'source':str(assets/'seed51'), 'original_epoch':78})
    shared = original.prepare(cfg)
    if (shared['revision'] != SOURCE or shared['config_hash'] != '3a93e11f5f4b'
            or json.loads(json.dumps(shared['protocol'])) != original_protocol):
        raise RuntimeError('seed51 원본 protocol과 불일치: 재개/새 학습 차단')
    # Exact archived seed49/50 rows, with input/design compatibility checked.
    previous = []
    for seed in (49, 50):
        with ZipFile(assets/f'seed{seed}.zip') as z:
            p = json.loads(z.read('protocol.json'))
            for key in ('input_hash', 'feature_sha256', 'split', 'data_stats',
                        'm3_audit', 'm4_weight_audit', 'propagation', 'regularization'):
                if p[key] != original_protocol[key]:
                    raise RuntimeError(f'seed{seed} 원본 호환성 불일치: {key}')
            for key, value in cfg.__dict__.items():
                if key not in ('seeds', 'out_dir', 'm2_reference', 'm4_reference') and p['config'][key] != value:
                    raise RuntimeError(f'seed{seed} 학습설정 불일치: {key}')
            for label in ORDER:
                row = json.loads(z.read(f'arms/gcn_{label.lower()}_lastweek_s{seed}.json'))
                validate_row(row, seed, label, shared['references']['metrics'])
                previous.append(row)
    protocol = {'version':VERSION, 'source_revision':SOURCE,
                'coordinator_sha256':sha(__file__), 'original_protocol':original_protocol,
                'seeds':list(ALL_SEEDS), 'execution_order':list(SEEDS),
                'reuse':[49,50], 'resume_seed51_epoch':78,
                'aggregation':'all80 only; mean and sample SD ddof=1; no significance claim'}
    out_dir.mkdir(parents=True, exist_ok=True)
    record = out_dir/'protocol.json'
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise RuntimeError('통합 실행 설정이 바뀌었습니다. 기존 파일 보존')
    original.base.io._atomic_json(record, protocol)
    return {'shared':shared, 'out':out_dir, 'previous':previous, 'original':original}


def run_new(ctx, seed, label):
    """Same forward, loss, sampling and optimizer; new identities for NEW seeds."""
    import torch
    o, p = ctx['original'], ctx['shared']
    if seed not in range(42,49) or label not in ORDER:
        raise ValueError('새 학습은 seed42~48만 허용')
    ident = o.base.RunIdentity(VERSION, f'gcn_{label.lower()}_lastweek', seed,
                               sha(ctx['out']/'protocol.json')[:12], SOURCE, p['input_hash'])
    path = ctx['out']/'arms'/f'{ident.model_id}_s{seed}.json'
    if path.exists():
        row = json.loads(path.read_text())
        validate_row(row, seed, label, p['references']['metrics'])
        if row['identity'] != asdict(ident):
            raise RuntimeError('완료 결과 실행 신원 불일치')
        print(f'[재사용] GCN seed{seed} {label}', flush=True)
        return
    store = o.SafeProgressStore(ctx['out']/'progress', ident)
    o.base.v3.set_seed(seed)
    model, weights = o.build_model(p, label)
    o.base.v3.set_seed(seed)
    try:
        history = o.execution.train(model, p, p['cfg'], ident.model_id, seed, store,
             row_weights=weights, negative_sampler=o.sample_uniform_negative_matrix)
        if [r['epoch'] for r in history] != list(range(1,301)):
            raise RuntimeError('300epoch 미완료: 평가 차단')
        metrics = o.base.fixed._evaluate(model, p)
        row = {'identity':asdict(ident), 'model':label, 'model_id':ident.model_id,
               'seed':seed, 'epochs':300, 'metrics':metrics, 'training_history':history,
               'diagnostics': model.representation_diagnostics() | model.training_gradient_diagnostics() |
               {'graph':'centered_nv' if o.ARMS[label][1] else 'binary',
                'row_weighted':o.ARMS[label][2], 'feature_sha256':p['feature_sha256']}}
        validate_row(row, seed, label, p['references']['metrics'])
        o.base.io._atomic_json(path, row)
        store.mark_complete(epoch=300, max_epoch=300, selection='none', result_path=str(path))
        o.compact_completed_checkpoint(store)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report(ctx):
    import pandas as pd
    o, p = ctx['original'], ctx['shared']
    rows = list(ctx['previous'])
    rows += [r for label in ORDER if (r := o.completed(p,label)) is not None]
    for path in sorted((ctx['out']/'arms').glob('*.json')):
        row = json.loads(path.read_text())
        if row['seed'] not in range(42,49) or row['model'] not in ORDER:
            raise RuntimeError('범위 밖 결과')
        validate_row(row,row['seed'],row['model'],p['references']['metrics'])
        rows.append(row)
    keys = [(r['seed'],r['model']) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('중복 결과')
    absolute = pd.DataFrame([{'seed':r['seed'],'model':r['model'],**r['metrics']} for r in rows])
    o.base.io._atomic_csv(ctx['out']/'absolute.csv',absolute)
    missing = [[s,m] for s in ALL_SEEDS for m in ORDER if (s,m) not in keys]
    status = {'complete':not missing, 'completed':len(rows), 'planned':80, 'missing':missing,
              'significance_claim':False, 'fixed_epoch':300}
    o.base.io._atomic_json(ctx['out']/'status.json',status)
    if not missing:
        metrics = list(p['references']['metrics'])
        summary = absolute.groupby('model')[metrics].agg(['mean','std'])
        summary.columns = [f'{m}__{stat}' for m,stat in summary.columns]
        o.base.io._atomic_csv(ctx['out']/'mean_std_10seeds.csv',summary.reindex(ORDER).reset_index())
    return status, rows


def run(ctx):
    for seed in SEEDS:
        for label in ORDER:
            print(f'[개인 GCN] seed{seed} {label}',flush=True)
            if seed == 51:
                ctx['original'].run_arm(ctx['shared'],label)
            else:
                run_new(ctx,seed,label)
            status,_ = report(ctx)
            print(f"[전체] {status['completed']}/80 완료",flush=True)
    return report(ctx)[0]


def export(ctx):
    status,rows = report(ctx)
    path = ctx['out']/('dh_gcn_final10_results' + ('' if status['complete'] else '_partial') + '.zip')
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for name in ('protocol.json','status.json','absolute.csv','mean_std_10seeds.csv'):
            f = ctx['out']/name
            if f.exists() and (name != 'mean_std_10seeds.csv' or status['complete']):
                z.write(f,name)
        for r in rows:
            z.writestr(f"arms/{r['model_id']}_s{r['seed']}.json",json.dumps(r,ensure_ascii=False))
    return str(path)
