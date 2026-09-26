"""Two new fits only: shared N/V M2 and M5, exact seed43 M1/M4 reuse."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m5_linear_nv_original_m4_lambda025_screen as prior
from clv_run_state import ProgressStore, RunIdentity, file_sha256
from clv_shared_nv_feature_model import SharedNVLightGCN, build_features

VERSION = 'shared-nv-feature-m2-m5-seed43-development-v1'
M2, M5 = 'm2_shared_nv_features_es', 'm5_shared_nv_original_masked_lambda025_es'
M4 = prior.lambda025.MODEL_ID
base, es = prior.base, prior.es
configure = prior.configure


def specs(settings=None):
    settings = settings or dict(alpha_n=.05, alpha_v=.05)
    return [dict(model_id=mid, role=role, kind='shared_nv_features', graph='binary',
                 weighted=role == 'M5', alpha_n=settings['alpha_n'], alpha_v=settings['alpha_v'])
            for mid, role in ((M2, 'M2'), (M5, 'M5'))]


def feature_hash(features):
    digest = hashlib.sha256()
    for key in sorted(k for k in features if k != 'diagnostics'):
        values = np.ascontiguousarray(features[key])
        digest.update(f'{key}:{values.dtype}:{values.shape}'.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare(report_path, out_dir, *, alpha_n=.05, alpha_v=.05):
    if not all(np.isfinite(x) and 0 < x <= .1 for x in (alpha_n, alpha_v)):
        raise ValueError('Both N/V strengths must be in (0,.1]; zero is not an improvement')
    cfg = configure(str(out_dir))
    prior.verified_anchors(report_path, cfg)  # No data load/training if reuse fails.
    prep = base._prepare(es.strength_cfg(cfg))
    audit, weights, diagnostic = prior.lambda025.previous.audit_weights(prep, cfg)
    prep.update(m4_weights=weights, m4_diagnostics=diagnostic)
    anchors = prior.verified_anchors(report_path, cfg, prep)
    data = prep['data']
    features = build_features(data['train'], n_users=data['n_users'], n_items=data['n_items'],
        q_n=prep['q_n'], q_v=prep['q_v'], valid=prep['clv_valid'],
        shrinkage=cfg.shrinkage_strength, bandwidth=cfg.basis_bandwidth)
    if not np.array_equal(features['keys'], np.asarray(data['pos_key'])):
        raise ValueError('Feature training pairs differ from the binary graph')
    settings = dict(alpha_n=float(alpha_n), alpha_v=float(alpha_v),
                    shrinkage=cfg.shrinkage_strength, bandwidth=cfg.basis_bandwidth)
    prep.update(anchors=[a for a in anchors if a['model_id'] in ('m1', M4)],
        source_report=str(report_path), source_report_sha256=prior.SOURCE_REPORT_SHA,
        screen_config=asdict(cfg), feature_settings=settings, features=features,
        features_sha256=feature_hash(features), feature_settings_sha=base._digest(settings))
    root = Path(out_dir)
    base.capacity.test10._atomic_csv(root/'m4_validity_audit.csv', audit)
    base.capacity.test10._atomic_json(root/'feature_diagnostic.json', features['diagnostics'])
    print('준비만 완료. M1·수정 원형 M4(λ=.25) 재사용; 새 학습 M2·M5 각 1개, seed43.', flush=True)
    print(json.dumps(dict(settings=settings, features=features['diagnostics'],
        max_epochs=cfg.epochs, final_test=False, holdout=False), ensure_ascii=False, indent=2))
    return cfg, prep, audit


def _build(prep, cfg):
    base.v3.set_seed(43)
    d, settings = prep['data'], prep['feature_settings']
    return SharedNVLightGCN(n_users=d['n_users'], n_items=d['n_items'],
        features=prep['features'], adj=d['adj'], id_dim=cfg.id_dim,
        axis_dim=cfg.history_axis_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
        alpha_n=settings['alpha_n'], alpha_v=settings['alpha_v']).to(base.v3.DEVICE)


def identity(prep, cfg, spec):
    result = prior.lambda025.previous.identity(prep, cfg, spec)
    result.update(version=VERSION, feature_settings=prep['feature_settings'],
                  features_sha256=prep['features_sha256'])
    for name in (Path(__file__).name, 'clv_shared_nv_feature_model.py'):
        result['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    return json.loads(json.dumps(result))


def save(arms, cfg, prep):
    metrics = {a['model_id']: a['metrics'] for a in arms}
    absolute = pd.DataFrame([dict(seed=43, model_id=a['model_id'], origin=a['origin'],
        selected_epoch=a['selected_epoch'], stopped_epoch=a['stopped_epoch'], **a['metrics']) for a in arms])
    rows = []
    for model_id, reference in ((M4, 'm1'), (M2, 'm1'), (M5, 'm1'), (M5, M4), (M5, M2)):
        if model_id not in metrics or reference not in metrics:
            continue
        for key, value in metrics[model_id].items():
            ref = metrics[reference].get(key)
            if isinstance(value, (int, float)) and isinstance(ref, (int, float)):
                rows.append(dict(seed=43, model_id=model_id, reference=reference, metric=key,
                    value=value, reference_value=ref, delta=value-ref,
                    relative_change_pct=100*(value/ref-1) if ref else np.nan))
    reading = dict(complete=all(k in metrics for k in ('m1', M4, M2, M5)),
                   significance_claim=False, scope='repeatedly exposed single development seed')
    for model_id in (M2, M5):
        if model_id in metrics:
            reading[model_id] = dict(
                accuracy_guard_vs_m1=all(metrics[model_id][k] >= .99*metrics['m1'][k] for k in base.ACCURACY),
                both_economic_at10_above_m1=all(metrics[model_id][k] > metrics['m1'][k] for k in es.fixed.PRIMARY))
    if M5 in metrics:
        reading[M5]['both_economic_at10_above_m4'] = all(metrics[M5][k] > metrics[M4][k] for k in es.fixed.PRIMARY)
    curve = pd.DataFrame([dict(seed=43, model_id=a['model_id'], epoch=r['epoch'], **r['metrics'])
                          for a in arms for r in a['curve']])
    diagnostics = pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r.get('diagnostics', {}))
                                for a in arms for r in a['curve']])
    paths, root = {}, Path(cfg.out_dir)/'reports'
    for name, frame in (('absolute', absolute), ('comparison', pd.DataFrame(rows)),
                        ('curve', curve), ('diagnostics', diagnostics)):
        paths[name] = str(root/f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(
        code_version=VERSION, config=asdict(cfg), feature_settings=prep['feature_settings'],
        new_arms=specs(prep['feature_settings']),
        config_note='legacy config fields are retained for exact baseline reuse; new_arms and feature_settings define the new representation',
        feature_diagnostics=prep['features']['diagnostics'], features_sha256=prep['features_sha256'],
        selection=es.preflight(prior.lambda025.previous.prior.configure(cfg.out_dir))['selection'],
        primary='overall weighted hit@10 and weighted NDCG@10; M5 vs M4 and M1',
        guard='six overall Recall/NDCG metrics each >= .99 * M1',
        m2='both fixed historical q_N/q_V -> trainable shared representations; no q_C, no economic propagation',
        m4='validity-masked original, lambda=.25; all-train-row mean normalization',
        reading=reading, arms=arms, final_test=False, holdout=False,
        source_report=prep['source_report'], source_report_sha256=prep['source_report_sha256'],
        m4_diagnostics=prep['m4_diagnostics'], paths=paths))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    if (feature_hash(prep['features']) != prep['features_sha256'] or
            base._digest(prep['feature_settings']) != prep['feature_settings_sha']):
        raise ValueError('Prepared features/settings changed')
    anchors = prior.verified_anchors(prep['source_report'], cfg, prep)
    arms = [a for a in anchors if a['model_id'] in ('m1', M4)]
    if sorted(a['model_id'] for a in arms) != sorted(['m1', M4]):
        raise ValueError('Both exact baseline results required; no baseline fitting allowed')
    _, weights, diagnostic = prior.lambda025.previous.audit_weights(prep, cfg)
    if not diagnostic['original_invalid_extra_absent'] or not np.array_equal(weights, prep['m4_weights']):
        raise ValueError('Prepared M4 weights changed')
    for spec in specs(prep['feature_settings']):
        ident = identity(prep, cfg, spec)
        key = base._digest(ident)
        root = Path(cfg.out_dir)/'arms'/key
        path = root/'result.json'
        if path.is_file():
            arm = json.loads(path.read_text())
            if arm['identity'] != ident:
                raise ValueError('Cached arm identity mismatch')
            prior._selected_arm(arm, cfg, prep)
            print(f"[cached] {spec['model_id']}", flush=True)
        else:
            store = ProgressStore(root/'progress', RunIdentity(VERSION, spec['model_id'], 43,
                key, 'content:'+base._digest(ident['source_hashes']), prep['input_hash']))
            model = _build(prep, cfg)
            result = es._train(model, prep, cfg, spec, 43, store, root)
            arm = dict(**spec, seed=43, identity=ident, origin='trained_shared_nv_joint', **result)
            base.capacity.test10._atomic_json(path, arm)
            store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
                best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arms.append(arm)
        paths = save(arms, cfg, prep)  # Preserve completed M2 readout if M5 is interrupted.
    return paths
