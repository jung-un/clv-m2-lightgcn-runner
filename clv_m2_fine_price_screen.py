"""Dunnhumby development screen: one jointly trained N/V + fine-type price M2."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_shared_feature_residual_m2_screen as prior
from clv_m2_fine_price import FINE_AXIS, FinePriceLightGCN, build_fine_features
from clv_run_state import ProgressStore, RunIdentity, file_sha256

VERSION = 'm2-fine-type-price-nv-seed43-development-v1'
MODEL_ID = 'm2_fine_type_price_nv_es'
SETTINGS = dict(user_l2=.001, item_l2=.001,
    axis_l2=dict.fromkeys((*prior.AXES, FINE_AXIS), .001), bandwidth=.25)
base, es, source = prior.base, prior.es, prior.source
configure, selection, read_source = prior.configure, prior.selection, prior.read_source
feature_hash = prior.feature_hash


def spec():
    return dict(model_id=MODEL_ID, role='M2', kind='fine_type_price_nv',
                graph='binary', weighted=False)


def prepare(report_path, out_dir):
    cfg = configure(str(out_dir))
    target = Path(out_dir)/'reports/result.json'
    if target.is_file() and json.loads(target.read_text())['code_version'] != VERSION:
        raise ValueError('Output belongs to another experiment')
    read_source(report_path, cfg)
    prep = base._prepare(es.strength_cfg(cfg))
    d, axes = prep['data'], prep['axes']
    if prep['base_cfg']['DATASET'] != 'dunnhumby':
        raise ValueError('This screen is fixed to Dunnhumby; the model also supports H&M')
    features = build_fine_features(d['train'], dataset='dunnhumby',
        n_users=d['n_users'], n_items=d['n_items'], q_n=axes['q_n'], q_v=axes['q_v'],
        n_valid=axes['activity_valid'], v_valid=axes['value_valid'],
        bandwidth=SETTINGS['bandwidth'])
    features['keys'] = np.asarray(d['pos_key'], np.int64).copy()
    prep.update(features=features, features_sha256=feature_hash(features),
        feature_settings=json.loads(json.dumps(SETTINGS)), screen_config=asdict(cfg),
        source_report=str(report_path), source_report_sha256=source.SOURCE_SHA)
    _, prep['anchors'] = read_source(report_path, cfg, prep)
    prior.removal.shared.movement._verify_new_item_truth(prep)
    base.capacity.test10._atomic_json(Path(out_dir)/'feature_diagnostic.json', features['diagnostics'])
    print('학습 전 확인 완료: 기존 M1 재사용, 새 M2 한 arm만 학습.', flush=True)
    print('사용자 N/V와 상품 가격의 세 좌표를 같은 layer0·plain BPR로 공동학습.', flush=True)
    print('세부 유형은 원시 유형 ID가 아니라 유형 내 가격 백분위에만 사용.', flush=True)
    print(json.dumps(features['diagnostics'][FINE_AXIS], ensure_ascii=False, indent=2), flush=True)
    return cfg, prep


def build(prep, cfg):
    base.v3.set_seed(43)
    d = prep['data']
    return FinePriceLightGCN(n_users=d['n_users'], n_items=d['n_items'], adj=d['adj'],
        features=prep['features'], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        **{k: v for k, v in prep['feature_settings'].items() if k != 'bandwidth'}).to(base.v3.DEVICE)


def diagnose(model, prep, cfg, arm, epoch):
    result = prior.diagnose(model, prep, cfg, arm, epoch)
    d, rng = prep['data'], np.random.default_rng(4301)
    ix = rng.choice(len(d['tr_u']), min(8192, len(d['tr_u'])), replace=False)
    negative = base.components.m4_helpers.sample_uniform_negative_matrix(
        d['tr_u'][ix], d['tr_i'][ix], d['n_items'], d['pos_key'], rng, k=1).reshape(-1)
    u, p, n = [torch.as_tensor(a, device=base.v3.DEVICE, dtype=torch.long)
               for a in (d['tr_u'][ix], d['tr_i'][ix], negative)]
    pos, neg = model._pair_scores(u, p, n)
    bpr = torch.nn.functional.softplus(neg-pos).mean()
    gradient = torch.autograd.grad(bpr, model.fine_encoder.weight)[0]
    result[FINE_AXIS+'_bpr_gradient_norm'] = float(gradient.detach().double().norm())
    result[FINE_AXIS+'_active'] = bool(np.isfinite(result[FINE_AXIS+'_bpr_gradient_norm'])
        and (result[FINE_AXIS+'_bpr_gradient_norm'] > 0 if epoch == 0
             else result[FINE_AXIS+'_mean_norm'] > 0))
    return result


def identity(prep, cfg):
    ident = prior.identity(prep, cfg)
    ident.update(version=VERSION, feature_settings=SETTINGS,
        features_sha256=prep['features_sha256'])
    for name in (Path(__file__).name, 'clv_m2_fine_price.py'):
        ident['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    return json.loads(json.dumps(ident))


def save(arms, cfg, prep, extra):
    m1, m2 = arms
    ref_curve = {r['epoch']: r['metrics'] for r in m1['curve']}
    rows = []
    for epoch, values, reference in [(None, m2['metrics'], m1['metrics'])]+[
        (r['epoch'], r['metrics'], ref_curve[r['epoch']])
        for r in m2['curve'] if r['epoch'] in ref_curve]:
        if values.keys() != reference.keys():
            raise ValueError('Metric sets differ from matched M1')
        for metric, value in values.items():
            rows.append(dict(seed=43, model_id=MODEL_ID, reference='m1', epoch=epoch,
                comparison='independently_selected' if epoch is None else 'same_epoch',
                selected_epoch=m2['selected_epoch'], reference_selected_epoch=m1['selected_epoch'],
                metric=metric, value=value, reference_value=reference[metric],
                delta=value-reference[metric],
                relative_change_pct=100*(value/reference[metric]-1) if reference[metric] else np.nan))
    comparison = pd.DataFrame(rows)
    frames = dict(absolute=pd.DataFrame([dict(seed=43, model_id=a['model_id'], origin=a['origin'],
        selected_epoch=a['selected_epoch'], stopped_epoch=a['stopped_epoch'], **a['metrics']) for a in arms]),
        comparison=comparison[comparison.epoch.isna()],
        same_epoch_comparison=comparison[comparison.epoch.notna()],
        curve=pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r['metrics'])
            for a in arms for r in a['curve']]),
        diagnostics=pd.DataFrame([dict(epoch=r['epoch'], **r['diagnostics'])
            for r in m2['training']['history'] if 'diagnostics' in r]), **extra)
    reading = dict(complete=True, significance_claim=False,
        accuracy_guard_vs_m1=all(m2['metrics'][k] >= .99*m1['metrics'][k] for k in base.ACCURACY),
        both_economic_at10_above_m1=all(m2['metrics'][k] > m1['metrics'][k] for k in es.fixed.PRIMARY))
    paths, root = {}, Path(cfg.out_dir)/'reports'
    for name, frame in frames.items():
        paths[name] = str(root/f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(code_version=VERSION, config=asdict(cfg),
        new_arms=[spec()], new_fit_count=1, feature_settings=SETTINGS, selection=selection(cfg),
        config_note='inherited rho/history/M4 fields unused; no dataset-specific coefficient chosen yet',
        representation='layer0: user ID+W_N phi_N+W_V phi_V; item ID+global/coarse/fine-type-conditioned price features',
        regularization='sampled ID L2/B + five separate shared matrix L2 terms; no auxiliary loss',
        clv_scope='historical N/V component M2; N is transaction activity, not SKU repeat; q_C not separately input',
        feature_diagnostics=prep['features']['diagnostics'], features_sha256=prep['features_sha256'],
        task='new-to-user; train pairs excluded; MIN_ITEM_INTER=1; binary graph; uniform K=1; plain BPR',
        split='historical_development_days_684_690', final_test=False, holdout=False,
        reading=reading, arms=arms, source_report=prep['source_report'], source_report_sha256=source.SOURCE_SHA,
        limits='one repeatedly exposed development seed; no significance/generalization/CLV attribution',
        diagnostic_note='selected-checkpoint axis removal is inference-only; not retrained causal ablation or matched M1',
        paths=paths))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    if prep['feature_settings'] != SETTINGS or feature_hash(prep['features']) != prep['features_sha256']:
        raise ValueError('Prepared features/settings changed')
    _, anchors = read_source(prep['source_report'], cfg, prep)
    ident = identity(prep, cfg)
    key = base._digest(ident)
    root = Path(cfg.out_dir)/'arms'/key
    path = root/'result.json'
    model = build(prep, cfg)
    if path.is_file():
        arm = json.loads(path.read_text())
        if arm['identity'] != ident or file_sha256(arm['checkpoint']) != arm['checkpoint_sha256']:
            raise ValueError('Completed identity/checkpoint changed')
        source.check_selected(arm, cfg)
        model.load_state_dict(torch.load(arm['checkpoint'], map_location='cpu', weights_only=False)['model_state'])
        print('[cached] 완료 M2 재사용; 추가 학습 없음', flush=True)
    else:
        store = ProgressStore(root/'progress', RunIdentity(VERSION, MODEL_ID, 43, key,
            'content:'+base._digest(ident['source_hashes']), prep['input_hash']))
        result = es._train(model, prep, cfg, spec(), 43, store, root, diagnose=diagnose)
        arm = dict(**spec(), seed=43, identity=ident, origin='trained_fine_type_price_m2', **result)
        base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
    extra = prior.selected_diagnostic(model, prep)
    full = extra['removal_metrics'].set_index('view').loc['full']
    for metric, value in arm['metrics'].items():
        if not np.isclose(full[metric], value, rtol=1e-5, atol=1e-8):
            raise RuntimeError(f'Selected checkpoint metric readback differs: {metric}')
    return save(anchors+[arm], cfg, prep, extra)
