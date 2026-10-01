"""Dunnhumby seed43 M2 with reliability applied after the price basis."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m2_support_backoff_price_screen as previous
from clv_m2_support_backoff_price import (
    AXES,
    SupportBackoffPriceLightGCN,
    build_support_backoff_features,
)
from clv_run_state import ProgressStore, RunIdentity, file_sha256

VERSION = 'm2-reliability-gated-price-nv-seed43-development-v1'
MODEL_ID = 'm2_reliability_gated_price_nv_es'
SETTINGS = dict(previous.SETTINGS, reliability_after_basis=True)
base, es, source = previous.base, previous.es, previous.source
configure, selection, read_source = previous.configure, previous.selection, previous.read_source
feature_hash = previous.feature_hash


def spec():
    return dict(model_id=MODEL_ID, role='M2', kind='reliability_gated_price_nv',
                graph='binary', weighted=False)


def prepare(report_path, out_dir):
    cfg = configure(str(out_dir))
    target = Path(out_dir) / 'reports/result.json'
    if target.is_file() and json.loads(target.read_text())['code_version'] != VERSION:
        raise ValueError('Output belongs to another experiment')
    read_source(report_path, cfg)
    prep = base._prepare(es.strength_cfg(cfg))
    d, axes = prep['data'], prep['axes']
    if prep['base_cfg']['DATASET'] != 'dunnhumby':
        raise ValueError('This first screen is fixed to Dunnhumby seed43')
    features = build_support_backoff_features(
        d['train'], dataset='dunnhumby', n_users=d['n_users'], n_items=d['n_items'],
        q_n=axes['q_n'], q_v=axes['q_v'], n_valid=axes['activity_valid'],
        v_valid=axes['value_valid'], bandwidth=SETTINGS['bandwidth'],
        shrinkage=SETTINGS['shrinkage'], reliability_after_basis=True,
    )
    price_info = features['diagnostics']['item_support_backoff_price']
    if price_info.get('reliability_gate_after_basis') is not True:
        raise RuntimeError('Price reliability gate was not applied after the basis')
    features['keys'] = np.asarray(d['pos_key'], np.int64).copy()
    prep.update(
        features=features,
        features_sha256=feature_hash(features),
        feature_settings=json.loads(json.dumps(SETTINGS)),
        screen_config=asdict(cfg),
        source_report=str(report_path),
        source_report_sha256=source.SOURCE_SHA,
    )
    _, prep['anchors'] = read_source(report_path, cfg, prep)
    previous.prior.removal.shared.movement._verify_new_item_truth(prep)
    base.capacity.test10._atomic_json(
        Path(out_dir) / 'feature_diagnostic.json', features['diagnostics'])
    print('준비 완료(학습 없음): 기존 M1 readout 재사용, 새 M2 한 arm만 학습.', flush=True)
    print('사용자 N/V 유지 + RBF 이후 구매고객 지지도를 곱한 상품 가격표현.', flush=True)
    print('binary graph·uniform K=1·plain BPR·한 optimizer·최대300epoch.', flush=True)
    print(json.dumps(price_info, ensure_ascii=False, indent=2), flush=True)
    return cfg, prep


def build(prep, cfg):
    base.v3.set_seed(43)
    d = prep['data']
    settings = dict(prep['feature_settings'])
    for key in ('bandwidth', 'shrinkage', 'reliability_after_basis'):
        settings.pop(key)
    return SupportBackoffPriceLightGCN(
        n_users=d['n_users'], n_items=d['n_items'], adj=d['adj'],
        features=prep['features'], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        **settings,
    ).to(base.v3.DEVICE)


def identity(prep, cfg):
    ident = base._identity(prep, es.strength_cfg(cfg, cfg.epochs), spec(), 43)
    ident.update(
        version=VERSION,
        config=asdict(cfg),
        selection=selection(cfg),
        feature_settings=SETTINGS,
        features_sha256=prep['features_sha256'],
        reference_report_sha256=source.SOURCE_SHA,
    )
    for name in (Path(__file__).name, 'clv_m2_support_backoff_price.py',
                 'clv_m2_support_backoff_price_screen.py'):
        ident['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    return json.loads(json.dumps(ident))


def save(arms, cfg, prep, extra):
    m1, m2 = arms
    reference_curve = {r['epoch']: r['metrics'] for r in m1['curve']}
    pairs = [(None, m2['metrics'], m1['metrics'])] + [
        (r['epoch'], r['metrics'], reference_curve[r['epoch']])
        for r in m2['curve'] if r['epoch'] in reference_curve
    ]
    rows = []
    for epoch, values, reference in pairs:
        if values.keys() != reference.keys():
            raise ValueError('Metric sets differ from matched M1')
        for metric, value in values.items():
            rows.append(dict(
                seed=43, model_id=MODEL_ID, reference=m1['model_id'], epoch=epoch,
                comparison='independently_selected' if epoch is None else 'same_epoch',
                selected_epoch=m2['selected_epoch'],
                reference_selected_epoch=m1['selected_epoch'], metric=metric,
                value=value, reference_value=reference[metric], delta=value-reference[metric],
                relative_change_pct=(100*(value/reference[metric]-1)
                                     if reference[metric] else np.nan),
            ))
    comparison = pd.DataFrame(rows)
    frames = dict(
        absolute=pd.DataFrame([dict(
            seed=43, model_id=a['model_id'], origin=a['origin'],
            selected_epoch=a['selected_epoch'], stopped_epoch=a['stopped_epoch'], **a['metrics'])
            for a in arms]),
        comparison=comparison[comparison.epoch.isna()],
        same_epoch_comparison=comparison[comparison.epoch.notna()],
        curve=pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r['metrics'])
                            for a in arms for r in a['curve']]),
        diagnostics=pd.DataFrame([dict(epoch=r['epoch'], **r['diagnostics'])
                                  for r in m2['training']['history'] if 'diagnostics' in r]),
        **extra,
    )
    ratios = {key: m2['metrics'][key]/m1['metrics'][key] if m1['metrics'][key] else None
              for key in base.ACCURACY}
    reading = dict(
        complete=True,
        significance_claim=False,
        accuracy_ratios_vs_m1=ratios,
        accuracy_guard_vs_m1=all(
            m2['metrics'][key] >= .99*m1['metrics'][key] for key in base.ACCURACY),
        both_economic_at10_above_m1=all(
            m2['metrics'][key] > m1['metrics'][key] for key in es.fixed.PRIMARY),
    )
    paths, root = {}, Path(cfg.out_dir) / 'reports'
    for name, frame in frames.items():
        paths[name] = str(root / f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root / 'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(
        code_version=VERSION,
        config=asdict(cfg),
        new_arms=[spec()],
        new_fit_count=1,
        feature_settings=SETTINGS,
        selection=selection(cfg),
        representation=(
            'layer0: user ID+N+V; item ID+one fine/coarse/global price basis '
            'multiplied by distinct-buyer reliability after the basis transform'),
        regularization=(
            'sampled ID L2/B + three separately penalized shared matrices; no auxiliary loss'),
        clv_scope='historical N/V component M2; N and V both retained; no separate q_C input',
        task=(
            'new-to-user; train pairs excluded; MIN_ITEM_INTER=1; binary graph; '
            'uniform K=1; plain BPR'),
        split='historical_development_days_684_690',
        final_test=False,
        holdout=False,
        reading=reading,
        feature_diagnostics=prep['features']['diagnostics'],
        features_sha256=prep['features_sha256'],
        arms=arms,
        source_report=prep['source_report'],
        source_report_sha256=source.SOURCE_SHA,
        limits=(
            'one repeatedly exposed development seed; no significance/generalization/CLV attribution'),
        paths=paths,
    ))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    if prep['feature_settings'] != SETTINGS or feature_hash(prep['features']) != prep['features_sha256']:
        raise ValueError('Prepared features/settings changed')
    _, anchors = read_source(prep['source_report'], cfg, prep)
    ident = identity(prep, cfg)
    key = base._digest(ident)
    root = Path(cfg.out_dir) / 'arms' / key
    result_path = root / 'result.json'
    model = build(prep, cfg)
    if result_path.is_file():
        arm = json.loads(result_path.read_text())
        if arm['identity'] != ident or file_sha256(arm['checkpoint']) != arm['checkpoint_sha256']:
            raise ValueError('Completed identity/checkpoint changed')
        source.check_selected(arm, cfg)
        model.load_state_dict(torch.load(
            arm['checkpoint'], map_location='cpu', weights_only=False)['model_state'])
        print('[cached] 완료 M2 재사용; 추가 학습 없음', flush=True)
    else:
        store = ProgressStore(root / 'progress', RunIdentity(
            VERSION, MODEL_ID, 43, key,
            'content:' + base._digest(ident['source_hashes']), prep['input_hash']))
        trained = es._train(
            model, prep, cfg, spec(), 43, store, root, diagnose=previous.diagnose)
        arm = dict(**spec(), seed=43, identity=ident,
                   origin='trained_reliability_gated_price_m2', **trained)
        base.capacity.test10._atomic_json(result_path, arm)
        store.mark_complete(
            epoch=trained['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=trained['selected_epoch'], result_path=str(result_path),
            checkpoint_path=trained['checkpoint'])
    extra = previous.prior.selected_diagnostic(model, prep)
    return save(anchors + [arm], cfg, prep, extra)

