"""H&M LightGCN seed51, eight frozen intervention arms, execution-only speedups.

Reference seed49 reports validate inputs, not seed51 performance. No model
selection, extra seeds, shuffle, intermediate test, or pretrained checkpoints.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import clv_m5_ab_adaptive_lastweek_seed49 as base
import clv_hm_remaining5_lastweek_seed50 as schema
import clv_ngcf_training_io as execution

VERSION = 'clv-hm-lightgcn-final8-seed51-v1'
SEED = 51
ARMS = {'M1': (False, False, False), 'M2': (True, False, False),
        'M3': (False, True, False), 'M4': (False, False, True),
        'M5_A': (True, True, False), 'M5_B': (True, False, True),
        'M5_C': (False, True, True), 'M5_D': (True, True, True)}
ORDER = tuple(ARMS)
CORE = schema.CORE
IDS = {'M1': base.hm.M1, 'M2': base.m2.MODEL_ID, 'M3': base.hm.M3,
       'M4': base.hm.M4_C, 'M5_C': base.hm.M5_C,
       'M5_A': schema.M5_A.replace('_s50', '_s51'),
       'M5_B': schema.M5_B.replace('_s50', '_s51'),
       'M5_D': schema.M5_D.replace('_s50', '_s51')}


def configure(*, out_dir=None, m2_reference=None, m4_reference=None):
    root = base.v3.default_out_dir('hm')
    old = base.configure(dataset='hm', out_dir=out_dir or root + '_final8_lastweek_seed51_v1',
                         m2_reference=m2_reference, m4_reference=m4_reference)
    return replace(old, seeds=(SEED,))


def validate_config(cfg):
    if cfg != configure(out_dir=cfg.out_dir, m2_reference=cfg.m2_reference,
                        m4_reference=cfg.m4_reference):
        raise ValueError('H&M seed51/300epoch/기존 고정 설정만 허용합니다')
    output = Path(cfg.out_dir).resolve()
    for source in (cfg.m2_reference, cfg.m4_reference):
        ref = Path(source).resolve()
        if output == ref or output in ref.parents or ref in output.parents:
            raise ValueError('새 출력과 참조 폴더는 겹칠 수 없습니다')


def load_references(cfg):
    validate_config(cfg)
    refs = {k: base._read_reference(getattr(cfg, k + '_reference'), 'hm_' + k)
            for k in ('m2', 'm4')}
    p2, p3 = (refs[k]['protocol'] for k in ('m2', 'm4'))
    for p in (p2, p3):
        schema._check_protocol(p, cfg, p2)
    for k in ('bandwidth', 'reliability_floor', 'shared_l2'):
        if p2['config'][k] != getattr(cfg, k):
            raise RuntimeError('M2 설정 불일치: ' + k)
    for k in schema.GRAPH_KEYS:
        if p3['config'][k] != getattr(base.hm.configure(), k):
            raise RuntimeError('M3 설정 불일치: ' + k)
    metrics = [c for c in refs['m2']['absolute'] if c not in ('model_id', 'seed')]
    if len(metrics) != 142:
        raise RuntimeError('설계 참조의 전체142지표 누락')
    return {'design': refs, 'metrics': metrics, 'audit': {k: r['audit'] for k, r in refs.items()}}


def prepare(cfg):
    started = time.perf_counter()
    refs = load_references(cfg)  # Fail before costly data loading; no seed50 dependency.
    p2, p3 = (refs['design'][k]['protocol'] for k in ('m2', 'm4'))
    print('seed49 원본은 설계/입력 검증용. seed51 8모형은 각각 새 초기화합니다.', flush=True)
    p = base._prepare_hm_inputs(cfg)
    data = p['data']
    base.hm.validate_split(data)
    if set(data['splits']) != {'test'} or data.get('loss_w') is not None:
        raise RuntimeError('test 단일 분할/기본 무가중 BPR 필요')
    if p['input_hash'] != p2['input_hash'] or data['data_stats'] != p2['data_stats']:
        raise RuntimeError('실제 입력/마지막주 분할 불일치')
    if p['feature_sha256'] != p2['feature_sha256'] or not np.array_equal(p['features']['keys'], data['pos_key']):
        raise RuntimeError('M2 특징/구매쌍 순서 불일치')
    if p['observed_weight_audit']['sha256'] != p3['m4_c_weight_audit']['sha256']:
        raise RuntimeError('M4 고정 가중치 불일치')
    inputs_seconds = time.perf_counter() - started
    print(f'[준비] 입력·CLV·M2·M4 {inputs_seconds:.1f}초', flush=True)
    t = time.perf_counter()
    keyed = base.hm.hm_graph._purchase_keyed(data['train'])
    signals = base.fixed.graph.centered_edge_signals(keyed, data['n_users'], data['n_items'])
    del keyed
    if not np.array_equal(signals['edge_users'] * data['n_items'] + signals['edge_items'], data['pos_key']):
        raise RuntimeError('M3 엣지/구매쌍 순서 불일치')
    p.update(signals=signals, q_value=p['q_v'].astype(float), q_activity=p['q_n'].astype(float))
    amount = p['item_amount_percentile']
    p['item_amount_percentile'] = amount.astype(float)
    gc = base.hm.configure(out_dir=str(Path(cfg.out_dir) / 'inputs'))
    graph = base.fixed.build_graph(p, gc)
    expected = p3['m3_audit']
    if set(graph['audit']) != set(expected) or any(not np.isclose(graph['audit'][k], v,
            rtol=1e-8, atol=1e-10) for k, v in expected.items()):
        raise RuntimeError('M3 고정 그래프 진단 불일치')
    p['observed_adj'] = base.v3.build_adj(signals['edge_users'], signals['edge_items'],
        graph['weights'].astype(np.float32), data['n_users'], data['n_items'])
    p['item_amount_percentile'] = amount
    code_hashes = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                  for n in (Path(__file__).name, 'clv_ngcf_training_io.py', 'clv_run_state.py')}
    protocol = {'code_version': VERSION, 'config': asdict(cfg), 'seeds': [SEED],
        'source_revision': p['revision'], 'code_sha256': code_hashes,
        'input_hash': p['input_hash'], 'data_stats': data['data_stats'], 'split': p2['split'],
        'feature_sha256': p['feature_sha256'],
        'dataset_initialization': p['protocol']['dataset_initialization'],
        'm3_config': {k: getattr(gc, k) for k in schema.GRAPH_KEYS},
        'm3_audit': graph['audit'], 'm3_beta': graph['beta'],
        'm4_weight_audit': p['observed_weight_audit'], 'reference_audit': refs['audit'],
        'models': IDS, 'interventions': ARMS, 'fixed': p3['fixed'], 'fixed_epoch': 300,
        'single_seed_only': True, 'significance_claim': False, 'unseen_confirmation_claim': False,
        'clv_attribution_claim': False,
        'reuse': 'seed49 reports validate frozen inputs only; own completed seed51 arms reused without fitting/evaluation',
        'initialization': 'fresh seed51 per arm; same original constructors; one joint Adam; no frozen/pretrained model',
        'execution': 'unchanged RNG draws, batches, graph, loss, Adam; binary-search final negative validation; fixed input tensors; tensor logs; local serialization then atomic synchronous Drive checkpoint EVERY epoch',
        'excluded': 'no extra seeds/shuffle/intermediate test/early stopping/AMP/TF32 switch/CSR/compile/embedding cache',
        'decision': 'seed51 descriptive result; do not tune on test or mix different settings into three-seed means'}
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    path = run_dir / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError('동일 출력 protocol 불일치: 덮어쓰지 않습니다')
    p.update(cfg=cfg, references=refs, protocol=protocol, config_hash=config_hash, run_dir=run_dir)
    p['base_cfg']['SEED_LIST'] = [SEED]
    base.io._atomic_json(path, protocol)
    base.io._atomic_json(run_dir / 'reference_audit.json', refs['audit'])
    timing = {'input_features_seconds': inputs_seconds, 'm3_seconds': time.perf_counter()-t,
              'total_seconds': time.perf_counter()-started}
    base.io._atomic_json(run_dir / 'preparation_timing.json', timing)
    for k in ('signals', 'q_value', 'q_activity'):
        p.pop(k, None)
    print(f'[준비] M3 {timing["m3_seconds"]:.1f}초 / 전체 {timing["total_seconds"]:.1f}초', flush=True)
    report(p)
    return p


def arm_inputs(p, label):
    if label not in ARMS:
        raise ValueError('승인된 8모형만 허용합니다')
    _, m3, m4 = ARMS[label]
    weights = p['observed_weights'] if m4 else None
    if m4 and weights is None:
        raise RuntimeError('M4 가중치 누락')
    return p['observed_adj'] if m3 else p['data']['adj'], weights


def identity(p, label):
    arm_inputs(p, label)
    return base.RunIdentity(VERSION, IDS[label], SEED, p['config_hash'], p['revision'], p['input_hash'])


def arm_path(p, label):
    return p['run_dir'] / 'arms' / f'{IDS[label]}_s{SEED}.json'


def completed(p, label):
    path = arm_path(p, label)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get('identity') != asdict(identity(p, label)) or row.get('epochs') != 300
            or row.get('model') != label or row.get('seed') != SEED or row.get('model_id') != IDS[label]
            or set(row.get('metrics', {})) != set(p['references']['metrics'])
            or not np.isfinite(list(row['metrics'].values())).all()):
        raise RuntimeError(label + ' 완료 결과 신원/300epoch/142지표 불일치')
    return row


class ProgressStore(schema.ProgressStore):
    def restore_epoch(self, model, optimizer, rng):
        if not self.latest_checkpoint.exists() and self.stage_path.exists():
            prior = json.loads(self.stage_path.read_text())
            if int(prior.get('epoch', 0)) > 1 or prior.get('checkpoint_sha256'):
                raise RuntimeError('진행 기록은 있으나 checkpoint가 없습니다. Drive 연결을 확인하세요')
        return super().restore_epoch(model, optimizer, rng)

    def save_epoch(self, model, optimizer, rng, **state):
        return execution.save_epoch_local_then_drive(self, model, optimizer, rng, state)


def build_model(p, label):
    adj, weights = arm_inputs(p, label)
    cfg, data = p['cfg'], p['data']
    base.v3.set_seed(SEED)
    if ARMS[label][0]:
        model = base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
            n_users=data['n_users'], n_items=data['n_items'], features=p['features'], adj=adj,
            id_dim=cfg.id_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
            shared_l2=cfg.shared_l2).to(base.v3.DEVICE)
    else:
        model = base.hm._build_model({'prepared': p, 'adjacency': adj}, cfg, SEED)
    return model, weights


def sample_uniform_negative_matrix(users, positives, n_items, positive_keys, rng, *, k):
    if k != 1:
        raise ValueError('고정 uniform K1만 허용합니다')
    negatives = base.v3.sample_negatives(users, positives, n_items, positive_keys, rng, 'uniform')
    keys = users.astype(np.int64) * int(n_items) + negatives
    positions = np.clip(np.searchsorted(positive_keys, keys), 0, len(positive_keys)-1)
    if (positive_keys[positions] == keys).any():
        raise RuntimeError('uniform 음성에 학습 관측상품이 남았습니다')
    return negatives[:, None]


def loss_with_tensor_logs(model, users, positives, negatives, weights):
    if isinstance(model, base.m2.DatasetAdaptiveOrthogonalNVLightGCN):
        return execution.loss_with_tensor_logs(model, users, positives, negatives, weights)
    # Preserve the plain model's original K=1 shapes and reduction order.
    user_z, item_z = model.propagated_embeddings()
    pos = (user_z[users] * item_z[positives]).sum(dim=1)
    neg = negatives[:, None]
    scores = (user_z[users, None, :] * item_z[neg]).sum(dim=2)
    rows = F.softplus(scores - pos[:, None]).mean(dim=1)
    bpr = rows.mean() if weights is None else (weights * rows).mean()
    return bpr + model.sampled_l2(users, positives, neg), (pos[:, None] > scores).float().mean().detach()


def run_arm(p, label):
    validate_config(p['cfg'])
    old = completed(p, label)
    if old is not None:
        print(f'[재사용] {label} seed51 — 재학습·재평가 없음', flush=True)
        return old
    if p['data'].get('loss_w') is not None:
        raise RuntimeError('M4 중복 가중 금지')
    if base.m2._feature_hash(p['features']) != p['feature_sha256']:
        raise RuntimeError('M2 특징 변경')
    _, weights = arm_inputs(p, label)
    if weights is not None and hashlib.sha256(np.asarray(weights, np.float32).tobytes()).hexdigest() != p['protocol']['m4_weight_audit']['sha256']:
        raise RuntimeError('M4 가중치 변경')
    store = ProgressStore(p['run_dir'] / 'progress', identity(p, label), heartbeat_interval_sec=300)
    model, weights = build_model(p, label)
    try:
        history = execution.train(model, p, p['cfg'], IDS[label], SEED, store, row_weights=weights,
            negative_sampler=sample_uniform_negative_matrix, loss_function=loss_with_tensor_logs)
        if [r['epoch'] for r in history] != list(range(1, 301)):
            raise RuntimeError('300epoch 미완료: test 평가 금지')
        print(f'[평가] {label} seed51 / 300epoch 마지막점', flush=True)
        t = time.perf_counter()
        metrics = base.fixed._evaluate(model, p)
        if set(metrics) != set(p['references']['metrics']) or not np.isfinite(list(metrics.values())).all():
            raise RuntimeError('최종142지표 불완전')
        diagnostic = {'graph': 'centered_nv' if ARMS[label][1] else 'binary',
            'row_weighted': ARMS[label][2], 'm2_intervention': ARMS[label][0],
            'graph_audit': p['protocol']['m3_audit'] if ARMS[label][1] else None,
            'weight_audit': p['protocol']['m4_weight_audit'] if ARMS[label][2] else None}
        if ARMS[label][0]:
            diagnostic.update(model.representation_diagnostics())
            diagnostic.update(model.training_gradient_diagnostics())
        row = {'identity': asdict(identity(p, label)), 'model': label, 'model_id': IDS[label],
            'seed': SEED, 'epochs': 300, 'metrics': metrics, 'training_history': history,
            'evaluation_seconds': time.perf_counter()-t, 'diagnostics': diagnostic}
        path = arm_path(p, label)
        base.io._atomic_json(path, row)
        store.mark_complete(epoch=300, max_epoch=300, selection='none', result_path=str(path))
        return row
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report(p):
    rows = [r for label in ORDER if (r := completed(p, label)) is not None]
    metrics = p['references']['metrics']
    absolute = pd.DataFrame([{'model': r['model'], 'model_id': r['model_id'], 'seed': SEED,
                              **r['metrics']} for r in rows], columns=['model', 'model_id', 'seed', *metrics])
    comparisons = []
    for candidate in rows:
        for reference in rows:
            if candidate['model'] == reference['model']:
                continue
            for metric in metrics:
                x, y = candidate['metrics'][metric], reference['metrics'][metric]
                comparisons.append({'seed': SEED, 'model': candidate['model'], 'reference': reference['model'],
                    'metric': metric, 'candidate_value': x, 'reference_value': y, 'delta': x-y,
                    'relative_change_pct': 100*(x/y-1) if y else None})
    tables = {'absolute': absolute, 'comparison': pd.DataFrame(comparisons, columns=[
        'seed', 'model', 'reference', 'metric', 'candidate_value', 'reference_value', 'delta', 'relative_change_pct']),
        'diagnostics': pd.DataFrame([{'model': r['model'], 'seed': SEED, **r['diagnostics']} for r in rows])}
    missing = [label for label in ORDER if label not in set(absolute.model)]
    reading = {'complete': not missing, 'missing': missing, 'seed': SEED, 'seed_count': 1,
        'completed_arm_count': len(rows), 'planned_arm_count': 8, 'fixed_epoch': 300,
        'significance_claim': False, 'clv_attribution_claim': False, 'unseen_confirmation_claim': False,
        'note': 'seed51 1시드 기술적 비교. seed49/50 혼합·대체 없음. 3시드 평균/CI는 원본 취합 후 별도 분석'}
    paths = {}
    for name, table in tables.items():
        paths[name] = str(p['run_dir'] / 'reports' / f'{name}.csv')
        base.io._atomic_csv(Path(paths[name]), table)
    paths['json'] = str(p['run_dir'] / 'reports/result.json')
    base.io._atomic_json(Path(paths['json']), {'protocol': p['protocol'], 'reading': reading, 'result_paths': paths})
    return {'complete': not missing, 'reading': reading, 'paths': paths, **tables}


def run(p):
    for index, label in enumerate(ORDER, 1):
        print(f'[{index}/8] H&M seed51 {label}', flush=True)
        run_arm(p, label)
        report(p)
    return report(p)


def timing_summary(p):
    """Training/save measurements only. No evaluation or extra fitting."""
    rows = []
    for label in ORDER:
        path = p['run_dir'] / 'progress' / 'timing' / f'{IDS[label]}_s{SEED}.csv'
        if not path.is_file():
            continue
        timing = pd.read_csv(path)
        if timing.empty:
            continue
        recent = timing.tail(5)
        epoch = int(timing.epoch.max())
        seconds = float((recent.train_seconds + recent.save_seconds).mean())
        rows.append({'model': label, 'last_saved_epoch': epoch,
            'recent_mean_train_seconds': float(recent.train_seconds.mean()),
            'recent_mean_save_seconds': float(recent.save_seconds.mean()),
            'remaining_training_hours_estimate': max(300-epoch, 0)*seconds/3600})
    return pd.DataFrame(rows)


def export_zip(p, destination, *, allow_partial=False):
    result = report(p)
    if not result['complete'] and not allow_partial:
        raise RuntimeError('8모형 미완료. 부분 회수는 allow_partial=True로 지정하세요')
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for path in result['paths'].values():
            archive.write(path, Path(path).name)
        for name in ('protocol.json', 'reference_audit.json', 'preparation_timing.json'):
            path = p['run_dir'] / name
            if path.is_file():
                archive.write(path, name)
        for label in ORDER:
            path = arm_path(p, label)
            if path.is_file():
                archive.write(path, 'arms/' + path.name)
        for path in sorted((p['run_dir'] / 'progress' / 'timing').glob('*.csv')):
            archive.write(path, 'timing/' + path.name)
    return str(destination)
