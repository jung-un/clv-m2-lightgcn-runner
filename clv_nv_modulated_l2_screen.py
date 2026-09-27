"""One new M2 fit: lower only shared N/V L2; reuse exact completed readouts."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

import clv_shared_nv_feature_screen as previous
from clv_run_state import ProgressStore, RunIdentity, file_sha256

VERSION = 'nv-modulated-id-m2-l2-1e4-seed43-development-v1'
MODEL_ID = 'm2_nv_modulated_id_shared_l2_1e4_es'
SOURCE_SHA = '19df7ea787ba428632f16c9a225ed1854b88b76d8fe9c65e086c50d35b36d43f'
OLD_SETTINGS = dict(eta_n=.05, eta_v=.05, shared_l2=.001, shrinkage=10., bandwidth=.25)
SETTINGS = dict(OLD_SETTINGS, shared_l2=.0001)
base, es = previous.base, previous.es
configure = previous.configure


def spec():
    return dict(previous.specs(SETTINGS)[0], model_id=MODEL_ID)


def selection(cfg):
    return es.preflight(previous.prior.lambda025.previous.prior.configure(cfg.out_dir))['selection']


def check_selected(arm, cfg):
    selected = es.replay(arm['curve'], cfg)
    if (arm['selected_epoch'] != selected['selected_epoch']
            or arm['stopped_epoch'] != selected['stopped_epoch']
            or arm['metrics'] != selected['selected_record']['metrics']):
        raise ValueError('Selected result differs from chronological early-stop replay')
    if not all(np.isfinite(arm['metrics'].get(k, np.nan)) for k in (*base.ACCURACY, *es.fixed.PRIMARY)):
        raise ValueError('Required performance metrics missing/nonfinite')


def read_source(path, cfg, prep=None):
    """Readouts only: no old weights are loaded, frozen, or needed for fresh training."""
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            payload = archive.read('result.json')
    else:
        payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError('Exact completed N/V-ID seed43 result.json required; no fallback training')
    report = json.loads(payload)
    expected = json.loads(json.dumps(asdict(cfg)))
    ignored = {'out_dir', 'reuse_dirs'}  # Paths only, never numerical experiment settings.
    if (report['code_version'] != previous.VERSION or report['final_test'] is not False
            or report['holdout'] is not False or report['selection'] != selection(cfg)
            or report['feature_settings'] != OLD_SETTINGS
            or {k: v for k, v in report['config'].items() if k not in ignored}
            != {k: v for k, v in expected.items() if k not in ignored}):
        raise ValueError('Reference protocol/settings differ; no training started')
    anchors = []
    for mid in ('m1', previous.M2):
        matches = [a for a in report['arms'] if a['model_id'] == mid and a['seed'] == 43]
        if len(matches) != 1:
            raise ValueError(f'Exactly one completed seed43 {mid} readout required')
        check_selected(matches[0], cfg)
        anchors.append(dict(matches[0], origin='reused_exact_nv_modulated_report_readout'))
    identity = anchors[1]['identity']
    if (identity['split'] != 'historical_development_days_684_690'
            or identity['feature_settings'] != OLD_SETTINGS or anchors[1]['weighted']):
        raise ValueError('Reference M2 identity mismatch')
    for name, digest in identity['source_hashes'].items():
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'Reference implementation changed: {name}; review before training')
    if prep is not None and (identity['input_hash'] != prep['input_hash']
            or report['features_sha256'] != prep['features_sha256']):
        raise ValueError('Training input/features differ from the reference experiment')
    return report, anchors


def prepare(report_path, out_dir):
    cfg = configure(str(out_dir))
    target = Path(out_dir)/'reports/result.json'
    if target.is_file() and json.loads(target.read_text())['code_version'] != VERSION:
        raise ValueError('Output belongs to another experiment; choose a new directory')
    read_source(report_path, cfg)  # Fail before large data preparation, never fit a missing baseline.
    prep = base._prepare(es.strength_cfg(cfg))
    d = prep['data']
    features = previous.build_features(d['train'], n_users=d['n_users'], n_items=d['n_items'],
        q_n=prep['q_n'], q_v=prep['q_v'], valid=prep['clv_valid'],
        shrinkage=cfg.shrinkage_strength, bandwidth=cfg.basis_bandwidth)
    if not np.array_equal(features['keys'], np.asarray(d['pos_key'])):
        raise ValueError('N/V training pairs differ from the binary graph')
    prep.update(features=features, features_sha256=previous.feature_hash(features),
        feature_settings=dict(SETTINGS), feature_settings_sha=base._digest(SETTINGS),
        screen_config=asdict(cfg), source_report=str(report_path), source_report_sha256=SOURCE_SHA)
    _, anchors = read_source(report_path, cfg, prep)
    prep['anchors'] = anchors
    base.capacity.test10._atomic_json(Path(out_dir)/'feature_diagnostic.json', features['diagnostics'])
    print('준비 완료(학습 없음). 기존 M1·M2 결과 재사용; 새 학습은 M2 하나, seed43.', flush=True)
    print('변경: 공유 N/V L2 .001 → .0001. ID L2=.001, eta_N=eta_V=.05, 최대300/기존 조기종료 유지.', flush=True)
    print('M4/M5 학습·손실 가중·H&M·추가 시드 없음. 최종 test/holdout 없음.', flush=True)
    return cfg, prep


def identity(prep, cfg):
    result = previous.identity(prep, cfg, spec())
    result.update(version=VERSION, baseline_report_sha256=SOURCE_SHA)
    result['protocol']['epochs'] = cfg.epochs
    result['source_hashes'][Path(__file__).name] = file_sha256(__file__)
    return result


def comparisons(arms, *, matched=False):
    indexed = {a['model_id']: a for a in arms}
    rows = []
    for mid, ref in ((previous.M2, 'm1'), (MODEL_ID, 'm1'), (MODEL_ID, previous.M2)):
        left, right = indexed[mid], indexed[ref]
        pairs = [(None, left['metrics'], right['metrics'])]
        if matched:
            other = {r['epoch']: r['metrics'] for r in right['curve']}
            pairs = [(r['epoch'], r['metrics'], other[r['epoch']])
                     for r in left['curve'] if r['epoch'] in other]
        for epoch, values, references in pairs:
            if values.keys() != references.keys():
                raise ValueError('Performance metric sets differ from reference')
            for key, value in values.items():
                baseline = references[key]
                rows.append(dict(seed=43, model_id=mid, reference=ref, metric=key,
                    comparison='same_epoch' if matched else 'independently_selected',
                    epoch=epoch, selected_epoch=left['selected_epoch'],
                    reference_selected_epoch=right['selected_epoch'], value=value,
                    reference_value=baseline, delta=value-baseline,
                    relative_change_pct=100*(value/baseline-1) if baseline else np.nan))
    return pd.DataFrame(rows)


def save(arms, cfg, prep):
    metrics = {a['model_id']: a['metrics'] for a in arms}
    absolute = pd.DataFrame([dict(seed=43, model_id=a['model_id'], origin=a['origin'],
        selected_epoch=a['selected_epoch'], stopped_epoch=a['stopped_epoch'], **a['metrics']) for a in arms])
    curve = pd.DataFrame([dict(seed=43, model_id=a['model_id'], epoch=r['epoch'], **r['metrics'])
                         for a in arms for r in a['curve']])
    diagnostics = pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r['diagnostics'])
        for a in arms for r in a.get('training', {}).get('history', a['curve']) if 'diagnostics' in r])
    reading = dict(complete=True, significance_claim=False,
        scope='one repeatedly exposed development seed; no CLV attribution/generalization claim')
    for mid in (previous.M2, MODEL_ID):
        ratios = {k: metrics[mid][k]/metrics['m1'][k] for k in base.ACCURACY}
        reading[mid] = dict(accuracy_ratios_vs_m1=ratios,
            accuracy_guard_vs_m1=all(v >= .99 for v in ratios.values()),
            both_economic_at10_above_m1=all(metrics[mid][k] > metrics['m1'][k] for k in es.fixed.PRIMARY))
    reading['new_m2_both_economic_at10_above_previous_m2'] = all(
        metrics[MODEL_ID][k] > metrics[previous.M2][k] for k in es.fixed.PRIMARY)
    paths, root = {}, Path(cfg.out_dir)/'reports'
    for name, frame in (('absolute', absolute), ('comparison', comparisons(arms)),
                        ('same_epoch_comparison', comparisons(arms, matched=True)),
                        ('curve', curve), ('diagnostics', diagnostics)):
        paths[name] = str(root/f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(
        code_version=VERSION, config=asdict(cfg), new_arms=[spec()], new_fit_count=1,
        config_note='inherited M4/legacy fields are unused: new_arms alone defines training; no M4/M5 fit or row weights',
        feature_settings=prep['feature_settings'], reference_feature_settings=OLD_SETTINGS,
        regularization=dict(id_coefficient=cfg.pref_reg, shared_nv_coefficient=SETTINGS['shared_l2'],
            formula='pref_reg * sampled_ID_squared_sum / batch_size + shared_l2 * shared_NV_squared_sum'),
        selection=selection(cfg), split='historical_development_days_684_690',
        task='new-to-user; train pairs excluded; MIN_ITEM_INTER=1; binary graph; uniform K=1; plain BPR',
        primary='overall weighted hit@10 and weighted NDCG@10 vs M1 and previous M2',
        guard='six overall Recall/NDCG each >= .99 * selected M1; high-CLV secondary',
        m2='unchanged N/V-conditioned ID; both axes jointly trained; no q_C or external reranking',
        same_epoch_note='intersection of observed epochs only; no interpolation or extra reference training',
        observed_epochs={a['model_id']: [r['epoch'] for r in a['curve']] for a in arms},
        probe_note='fixed TRAIN probe, not Top-10 attribution; initial probes also stored under arms',
        reading=reading, arms=arms, features_sha256=prep['features_sha256'],
        feature_diagnostics=prep['features']['diagnostics'], final_test=False, holdout=False,
        source_report=prep['source_report'], source_report_sha256=SOURCE_SHA,
        reuse='exact report metrics/curves only; old checkpoints not loaded; no automatic baseline fitting', paths=paths))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    if (prep['feature_settings'] != SETTINGS or previous.feature_hash(prep['features']) != prep['features_sha256']
            or base._digest(prep['feature_settings']) != prep['feature_settings_sha']):
        raise ValueError('Prepared features/settings changed')
    _, anchors = read_source(prep['source_report'], cfg, prep)
    ident = identity(prep, cfg)
    key = base._digest(ident)
    root = Path(cfg.out_dir)/'arms'/key
    path = root/'result.json'
    if path.is_file():
        arm = json.loads(path.read_text())
        if arm['identity'] != ident:
            raise ValueError('Cached arm identity mismatch')
        check_selected(arm, cfg)
        print('[cached] 기존 완료 결과 재사용; 추가 학습 없음', flush=True)
    else:
        store = ProgressStore(root/'progress', RunIdentity(VERSION, MODEL_ID, 43, key,
            'content:'+base._digest(ident['source_hashes']), prep['input_hash']))
        model = previous._build(prep, cfg)
        result = es._train(model, prep, cfg, spec(), 43, store, root, diagnose=previous.diagnose)
        arm = dict(**spec(), seed=43, identity=ident, origin='trained_nv_l2_1e4_only', **result)
        base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return save(anchors+[arm], cfg, prep)
