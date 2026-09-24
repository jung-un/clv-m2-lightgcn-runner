"""Original M4 versus original M4 + linear N/V; no implicit formula repair."""
from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

import clv_linear_nv_affine_l2_screen as prior
from clv_run_state import ProgressStore, RunIdentity, file_sha256

es = prior.es
base = prior.base
VERSION = 'linear-nv-original-m4-seed43-development-v1'
M4 = 'm4_original_es'
M5 = 'm5_linear_nv_original_es'


def configure(out_dir):
    # The existing ES validator intentionally only accepts complementary M4.
    # Validate all shared settings through it, then change ONLY the M4 formula.
    return replace(prior.configure(out_dir), m4_mode='original')


def specs():
    return [dict(model_id=M4, role='M4', kind='id', graph='binary', weighted=True, rho=0.),
            dict(model_id=M5, role='M5', kind='history_fit', graph='binary',
                 weighted=True, rho=.05, condition='nv')]


def audit_weights(prep, cfg):
    """Report existing formulas verbatim; never repair/mask training weights here."""
    u = np.asarray(prep['data']['tr_u'], dtype=np.int64)
    i = np.asarray(prep['data']['tr_i'], dtype=np.int64)
    masks = dict(clv=np.asarray(prep['clv_valid'], bool)[u],
                 user_economic=np.asarray(prep['user_economic_valid'], bool)[u],
                 item_economic=np.asarray(prep['item_economic_valid'], bool)[i])
    valid = np.logical_and.reduce(list(masks.values()))
    rows = []
    original_weights = original_meta = None
    for mode in ('original', 'complementary'):
        weights, meta = base.weights_module.row_weights(prep, cfg, mode)
        raw = weights * meta['train_mean_raw_weight']
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Invalid M4 weights; no training permitted')
        groups = {'all': np.ones(len(u), bool), 'all_inputs_valid': valid,
                  'any_input_invalid': ~valid,
                  **{f'{name}_invalid': ~mask for name, mask in masks.items()}}
        for group, mask in groups.items():
            rows.append(dict(mode=mode, group=group, n_rows=int(mask.sum()),
                row_share=float(mask.mean()),
                raw_extra_sum=float((raw[mask]-1).sum()),
                rows_with_extra=int(np.count_nonzero(raw[mask] > 1 + 1e-10)),
                normalized_weight_share=float(weights[mask].sum()/weights.sum()),
                mean_raw=float(raw[mask].mean()) if mask.any() else None))
        if mode == 'original':
            original_weights, original_meta = weights, meta
    frame = pd.DataFrame(rows)
    bad = frame[(frame['mode']=='original') & (frame['group']=='any_input_invalid')]
    safe = int(bad.iloc[0].rows_with_extra) == 0
    return frame, original_weights, dict(original_meta,
        original_invalid_extra_absent=safe,
        invalid_item_negative_bin_rows=int(np.count_nonzero(np.asarray(prep['item_bin'])[i] < 0)),
        policy='preserve original formula; block training if invalid rows receive raw extra weight',
        note='Groups overlap. Weight mass is not gradient mass or performance attribution.')


def prepare(report_path, out_dir):
    cfg = configure(out_dir)
    anchor_cfg = prior.configure(out_dir)
    if not Path(report_path).is_file() or file_sha256(report_path) != prior.EXPECTED_REPORT_SHA:
        raise ValueError('Missing/changed original seed43 report; no training started')
    prep = base._prepare(es.strength_cfg(anchor_cfg))
    anchors = prior.verified_anchors(report_path, anchor_cfg, prep)
    audit, weights, diagnostic = audit_weights(prep, cfg)
    root = Path(out_dir)
    base.capacity.test10._atomic_csv(root/'m4_validity_audit.csv', audit)
    base.capacity.test10._atomic_json(root/'m4_validity_audit.json', diagnostic)
    prep.update(anchors=anchors, source_report=str(report_path),
                source_report_sha256=file_sha256(report_path), screen_config=asdict(cfg),
                m4_weights=weights, m4_diagnostics=diagnostic)
    print('진단만 완료. M1/보완 M4/기존 M5 재사용. 새 원형 M4/M5는 각각 최대300 epoch.', flush=True)
    print('학습 가능:', diagnostic['original_invalid_extra_absent'], flush=True)
    return cfg, prep, audit


def identity(prep, cfg, spec):
    result = base._identity(prep, es.strength_cfg(cfg), spec, 43)
    result.update(version=VERSION, config=asdict(cfg),
        selection=es.preflight(prior.configure(cfg.out_dir))['selection'],
        baseline_report_sha256=prep['source_report_sha256'])
    for name in (Path(__file__).name, 'clv_linear_nv_affine_l2_screen.py',
                 'clv_history_linear_nv_model.py', 'lightgcn_clv_history_linear_nv.py',
                 'lightgcn_clv_history_linear_nv_early_stop.py'):
        result['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    return json.loads(json.dumps(result))


def save(arms, cfg, prep):
    metrics = {a['model_id']: a['metrics'] for a in arms}
    absolute = pd.DataFrame([dict(model_id=a['model_id'], seed=43,
        selected_epoch=a['selected_epoch'], stopped_epoch=a['stopped_epoch'],
        origin=a['origin'], **a['metrics']) for a in arms])
    pairs = [('m4','m1'), ('m5_linear_nv','m1'), ('m5_linear_nv','m4'),
             (M4,'m1'), (M4,'m4'), (M5,'m1'), (M5,M4), (M5,'m4'), (M5,'m5_linear_nv')]
    rows = []
    for mid, ref in pairs:
        if mid not in metrics or ref not in metrics:
            continue
        for key, value in metrics[mid].items():
            baseline = metrics[ref].get(key)
            if isinstance(value, (int,float)) and isinstance(baseline, (int,float)):
                rows.append(dict(seed=43, model_id=mid, reference=ref, metric=key,
                    value=value, reference_value=baseline, delta=value-baseline,
                    relative_change_pct=100*(value/baseline-1) if baseline else np.nan))
    comparison = pd.DataFrame(rows)
    reading = dict(complete=M4 in metrics and M5 in metrics, significance_claim=False,
                   scope='single development seed; all segments and @20/@50 must be reported')
    if M5 in metrics:
        reading['accuracy_guard_vs_m1'] = all(metrics[M5][k]>=.99*metrics['m1'][k] for k in base.ACCURACY)
        for ref in ('m1',M4,'m4','m5_linear_nv'):
            if ref in metrics:
                reading['both_economic_at10_above_'+ref] = all(metrics[M5][k]>metrics[ref][k] for k in es.fixed.PRIMARY)
    curve = pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r['metrics'])
                          for a in arms for r in a['curve']])
    root = Path(cfg.out_dir)/'reports'
    paths = {}
    for name, frame in [('absolute',absolute), ('comparison',comparison), ('curve',curve)]:
        paths[name] = str(root/f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(code_version=VERSION,
        config=asdict(cfg), selection=es.preflight(prior.configure(cfg.out_dir))['selection'],
        reading=reading, arms=arms, final_test=False, holdout=False,
        source_report=prep['source_report'], m4_diagnostics=prep['m4_diagnostics'], paths=paths))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    anchors = prior.verified_anchors(prep['source_report'], prior.configure(cfg.out_dir), prep)
    _, weights, diagnostic = audit_weights(prep, cfg)
    if not diagnostic['original_invalid_extra_absent']:
        raise RuntimeError('원형 M4의 무효 입력에 추가 가중치가 있습니다. 감사 ZIP을 먼저 확인하세요. 수식 수정/새 학습을 하지 않았습니다.')
    if not np.array_equal(weights, prep['m4_weights']):
        raise ValueError('Prepared weights changed')
    arms = list(anchors)
    for spec in specs():
        ident = identity(prep, cfg, spec)
        key = base._digest(ident)
        root = Path(cfg.out_dir)/'arms'/key
        path = root/'result.json'
        if path.is_file():
            arm = json.loads(path.read_text())
            selected = es.replay(arm['curve'], cfg)
            if (arm['identity'] != ident or arm['metrics'] != selected['selected_record']['metrics']
                or arm['selected_epoch'] != selected['selected_epoch'] or arm['stopped_epoch'] != selected['stopped_epoch']):
                raise ValueError('Cached result mismatch')
        else:
            store = ProgressStore(root/'progress', RunIdentity(VERSION,spec['model_id'],43,key,
                'content:'+base._digest(ident['source_hashes']),prep['input_hash']))
            build = es.fixed._build if spec['model_id']==M5 else base._build_model
            model = build(prep, es.strength_cfg(cfg), spec, 43)
            result = es._train(model, prep, cfg, spec, 43, store, root)
            arm = dict(**spec, seed=43, identity=ident, origin='trained_original_m4_screen', **result)
            base.capacity.test10._atomic_json(path, arm)
            store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
                best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arms.append(arm)
        paths = save(arms, cfg, prep)
    return paths
