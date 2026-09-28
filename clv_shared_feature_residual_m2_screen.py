"""One new Dunnhumby M2 fit, exact M1 readout reuse, no M4/M5/H&M fitting."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_nv_modulated_l2_screen as source
import clv_nv_modulated_checkpoint_diagnostic as removal
from clv_run_state import ProgressStore, RunIdentity, file_sha256
from clv_shared_feature_residual_m2 import AXES, SharedFeatureResidualLightGCN, build_features

VERSION = 'shared-feature-residual-m2-seed43-development-v1'
MODEL_ID = 'm2_shared_feature_residual_es'
SETTINGS = dict(user_l2=.001, item_l2=.001, axis_l2=dict.fromkeys(AXES, .001), bandwidth=.25)
base, es = source.base, source.es
configure, selection = source.configure, source.selection
feature_hash = source.previous.feature_hash


def spec():
    return dict(model_id=MODEL_ID, role='M2', kind='shared_feature_residual',
                graph='binary', weighted=False)


def read_source(path, cfg, prep=None):
    report, anchors = source.read_source(path, cfg)
    if prep is not None and anchors[1]['identity']['input_hash'] != prep['input_hash']:
        raise ValueError('Training input differs from exact reference; no baseline fitting')
    return report, [anchors[0]]  # Readouts only: no pretrained weights in the new M2.


def prepare(report_path, out_dir):
    cfg = configure(str(out_dir))
    target = Path(out_dir)/'reports/result.json'
    if target.is_file() and json.loads(target.read_text())['code_version'] != VERSION:
        raise ValueError('Output belongs to another experiment')
    read_source(report_path, cfg)
    prep = base._prepare(es.strength_cfg(cfg))
    d, axes = prep['data'], prep['axes']
    features = build_features(d['train'], n_users=d['n_users'], n_items=d['n_items'],
        q_n=axes['q_n'], q_v=axes['q_v'], n_valid=axes['activity_valid'],
        v_valid=axes['value_valid'], bandwidth=SETTINGS['bandwidth'])
    features['keys'] = np.asarray(d['pos_key'], np.int64).copy()
    prep.update(features=features, features_sha256=feature_hash(features),
        feature_settings=json.loads(json.dumps(SETTINGS)), screen_config=asdict(cfg),
        source_report=str(report_path), source_report_sha256=source.SOURCE_SHA)
    _, prep['anchors'] = read_source(report_path, cfg, prep)
    removal.shared.movement._verify_new_item_truth(prep)
    base.capacity.test10._atomic_json(Path(out_dir)/'feature_diagnostic.json', features['diagnostics'])
    print('준비 완료(학습 없음). 기존 M1 곡선 재사용; 새 M2 하나만 seed43으로 학습.', flush=True)
    print('사용자 N/V 유지, 상품 구매자 N 대신 전체/카테고리 내 가격 특징. 모두 layer0 내부 공동학습.', flush=True)
    print('ID64·2층·최대300, 25마다 평가·100 이후4회 미개선 중단. M3/M4/M5/H&M 추가 학습 없음.', flush=True)
    print(json.dumps(features['diagnostics'], ensure_ascii=False, indent=2), flush=True)
    return cfg, prep


def build(prep, cfg):
    base.v3.set_seed(43)
    d = prep['data']
    return SharedFeatureResidualLightGCN(n_users=d['n_users'], n_items=d['n_items'],
        adj=d['adj'], features=prep['features'], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        **{k: v for k, v in prep['feature_settings'].items() if k != 'bandwidth'}).to(base.v3.DEVICE)


def diagnose(model, prep, cfg, arm, epoch):
    d, rng = prep['data'], np.random.default_rng(4301)
    ix = rng.choice(len(d['tr_u']), min(8192, len(d['tr_u'])), replace=False)
    negative = base.components.m4_helpers.sample_uniform_negative_matrix(
        d['tr_u'][ix], d['tr_i'][ix], d['n_items'], d['pos_key'], rng, k=1).reshape(-1)
    u, p, n = [torch.as_tensor(a, device=base.v3.DEVICE, dtype=torch.long)
               for a in (d['tr_u'][ix], d['tr_i'][ix], negative)]
    pos, neg = model._pair_scores(u, p, n)
    bpr = torch.nn.functional.softplus(neg-pos).mean()
    parameters = [layer.weight for layer in model.encoders.values()]
    gradients = torch.autograd.grad(bpr, parameters)
    penalties = torch.autograd.grad(model.batch_l2(u, p, n), parameters)
    result = dict(model.representation_diagnostics(), probe_rows=len(ix), probe_bpr=float(bpr.detach()))
    for name, gradient, penalty in zip(AXES, gradients, penalties):
        result[name+'_bpr_gradient_norm'] = float(gradient.detach().double().norm())
        result[name+'_l2_gradient_norm'] = float(penalty.detach().double().norm())
    suffix = '_bpr_gradient_norm' if epoch == 0 else '_mean_norm'
    result['nv_activity_ok'] = bool(all(np.isfinite(v) for v in result.values())
        and all(result[name+suffix] > 0 for name in ('user_n', 'user_v')))
    print(f"[N/V probe ep{epoch}] active={result['nv_activity_ok']} | "
          f"N norm={result['user_n_mean_norm']:.3g}, V norm={result['user_v_mean_norm']:.3g}", flush=True)
    return result


def identity(prep, cfg):
    ident = base._identity(prep, es.strength_cfg(cfg, cfg.epochs), spec(), 43)
    ident.update(version=VERSION, config=asdict(cfg), selection=selection(cfg),
        feature_settings=SETTINGS, features_sha256=prep['features_sha256'],
        reference_report_sha256=source.SOURCE_SHA)
    for name in (Path(__file__).name, 'clv_shared_feature_residual_m2.py',
                 'lightgcn_clv_history_linear_nv_early_stop.py', 'clv_nv_modulated_l2_screen.py',
                 'clv_nv_modulated_checkpoint_diagnostic.py', 'clv_run_state.py',
                 'lightgcn_clv_joint_nv.py', 'lightgcn_clv_gatefree_lowdim.py'):
        ident['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    return json.loads(json.dumps(ident))


@torch.no_grad()
def selected_diagnostic(model, prep):
    # Reuse the existing rank/movement calculator. Here id_only means PRICE-ONLY
    # within this trained M2, not M1. All removals are inference-only diagnostics.
    views, metrics = {}, []
    disabled = dict(id_only=('user_n', 'user_v'), id_n=('user_v',),
                    id_v=('user_n',), full=())
    labels = dict(id_only='without_user_nv', id_n='without_user_v',
                  id_v='without_user_n', full='full')
    try:
        for view, axes in disabled.items():
            model.disabled_axes = axes
            views[view] = model.embeddings()[:2]
            metrics.append(dict(view=labels[view], **base.capacity._evaluate(model, prep)))
        removal.shared.movement._verify_new_item_truth(prep)
        tops = removal.rank_views(views, prep)
        truth, users, summary = removal.movements(tops, prep)
        for frame in (truth, users, summary):
            for column in ('view', 'reference'):
                frame[column] = frame[column].map(labels)
        recommendations = pd.concat([pd.DataFrame(dict(view=labels[name],
            user=np.repeat(prep['cache'].users, 50), rank=np.tile(np.arange(1, 51), len(top)),
            item=top.reshape(-1))) for name, top in tops.items()], ignore_index=True)
        return dict(removal_metrics=pd.DataFrame(metrics), truth_movements=truth,
            user_movements=users, movement_summary=summary, recommendations=recommendations)
    finally:
        model.disabled_axes = ()


def save(arms, cfg, prep, extra):
    m1, m2 = arms
    pairs = [(None, m2['metrics'], m1['metrics'])]
    reference = {r['epoch']: r['metrics'] for r in m1['curve']}
    pairs += [(r['epoch'], r['metrics'], reference[r['epoch']])
              for r in m2['curve'] if r['epoch'] in reference]
    rows = []
    for epoch, values, ref in pairs:
        if values.keys() != ref.keys():
            raise ValueError('Metric sets differ from the reference')
        for metric, value in values.items():
            rows.append(dict(seed=43, model_id=MODEL_ID, reference='m1', epoch=epoch,
                comparison='independently_selected' if epoch is None else 'same_epoch',
                selected_epoch=m2['selected_epoch'], reference_selected_epoch=m1['selected_epoch'],
                metric=metric, value=value, reference_value=ref[metric], delta=value-ref[metric],
                relative_change_pct=100*(value/ref[metric]-1) if ref[metric] else np.nan))
    comparison = pd.DataFrame(rows)
    frames = dict(absolute=pd.DataFrame([dict(seed=43, model_id=a['model_id'], origin=a['origin'],
        selected_epoch=a['selected_epoch'], stopped_epoch=a['stopped_epoch'], **a['metrics']) for a in arms]),
        comparison=comparison[comparison.epoch.isna()],
        same_epoch_comparison=comparison[comparison.epoch.notna()],
        curve=pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r['metrics'])
            for a in arms for r in a['curve']]),
        diagnostics=pd.DataFrame([dict(epoch=r['epoch'], **r['diagnostics'])
            for r in m2['training']['history'] if 'diagnostics' in r]), **extra)
    ratios = {k: m2['metrics'][k]/m1['metrics'][k] if m1['metrics'][k] else None
              for k in base.ACCURACY}
    reading = dict(complete=True, significance_claim=False, accuracy_ratios_vs_m1=ratios,
        accuracy_guard_vs_m1=all(m2['metrics'][k] >= .99*m1['metrics'][k] for k in base.ACCURACY),
        both_economic_at10_above_m1=all(m2['metrics'][k] > m1['metrics'][k] for k in es.fixed.PRIMARY))
    paths, root = {}, Path(cfg.out_dir)/'reports'
    for name, frame in frames.items():
        paths[name] = str(root/f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(code_version=VERSION, config=asdict(cfg),
        new_arms=[spec()], new_fit_count=1, feature_settings=SETTINGS, selection=selection(cfg),
        config_note='inherited rho/history/M4 fields unused; only new_arms defines fitting',
        representation='layer0: user ID residual + W_N phi_N + W_V phi_V; item ID residual + W_P phi_P + W_PC phi_PC',
        regularization='sampled raw ID L2/B + four separately penalized shared matrices; no auxiliary loss',
        clv_scope='historical N/V component M2; no separate q_C input; N is transaction activity, not SKU repeat',
        feature_diagnostics=prep['features']['diagnostics'], features_sha256=prep['features_sha256'],
        task='new-to-user; train pairs excluded; MIN_ITEM_INTER=1; binary graph; uniform K=1; plain BPR',
        split='historical_development_days_684_690', final_test=False, holdout=False,
        reading=reading, arms=arms, source_report=prep['source_report'], source_report_sha256=source.SOURCE_SHA,
        limits='one repeatedly exposed development seed; no significance/generalization/CLV attribution',
        diagnostic_note='selected-checkpoint removals retain item price and trained ID; not M1 or causal attribution. M1 Top-K unavailable: no M1 retraining; same-epoch comparison is observed intersection only.',
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
        arm = dict(**spec(), seed=43, identity=ident, origin='trained_shared_feature_residual', **result)
        base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
    extra = selected_diagnostic(model, prep)
    full = extra['removal_metrics'].set_index('view').loc['full']
    for metric, value in arm['metrics'].items():
        if not np.isclose(full[metric], value, rtol=1e-5, atol=1e-8):
            raise RuntimeError(f'Selected checkpoint metric readback differs: {metric}')
    return save(anchors+[arm], cfg, prep, extra)
