"""N/V-conditioned ID representation; exact seed43 M1/M4 reuse, two new arms."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m5_linear_nv_original_m4_lambda025_screen as prior
from clv_run_state import ProgressStore, RunIdentity, file_sha256
from clv_shared_nv_feature_model import NVModulatedLightGCN, build_features

VERSION = 'nv-modulated-id-m2-m5-seed43-development-v3'
M2, M5 = 'm2_nv_modulated_id_es', 'm5_nv_modulated_id_original_masked_lambda025_es'
M4 = prior.lambda025.MODEL_ID
base, es = prior.base, prior.es
configure = prior.configure


def specs(settings=None):
    settings = settings or dict(eta_n=.05, eta_v=.05, shared_l2=.001)
    return [dict(model_id=mid, role=role, kind='nv_modulated_id', graph='binary',
                 weighted=role == 'M5', eta_n=settings['eta_n'], eta_v=settings['eta_v'],
                 shared_l2=settings['shared_l2'])
            for mid, role in ((M2, 'M2'), (M5, 'M5'))]


def feature_hash(features):
    digest = hashlib.sha256()
    for key in sorted(k for k in features if k != 'diagnostics'):
        values = np.ascontiguousarray(features[key])
        digest.update(f'{key}:{values.dtype}:{values.shape}'.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare(report_path, out_dir, *, eta_n=.05, eta_v=.05, shared_l2=.001):
    if not all(np.isfinite(x) and 0 < x <= .1 for x in (eta_n, eta_v)):
        raise ValueError('Both N/V strengths must be in (0,.1]; zero is not an improvement')
    if not np.isfinite(shared_l2) or shared_l2 <= 0:
        raise ValueError('Positive finite shared L2 required for this screen')
    cfg = configure(str(out_dir))
    existing = Path(out_dir)/'reports/result.json'
    if existing.is_file() and json.loads(existing.read_text()).get('code_version') != VERSION:
        raise ValueError('Output contains another experiment version; choose a new directory')
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
    settings = dict(eta_n=float(eta_n), eta_v=float(eta_v), shared_l2=float(shared_l2),
                    shrinkage=cfg.shrinkage_strength, bandwidth=cfg.basis_bandwidth)
    prep.update(anchors=[a for a in anchors if a['model_id'] in ('m1', M4)],
        source_report=str(report_path), source_report_sha256=prior.SOURCE_REPORT_SHA,
        screen_config=asdict(cfg), feature_settings=settings, features=features,
        features_sha256=feature_hash(features), feature_settings_sha=base._digest(settings))
    root = Path(out_dir)
    base.capacity.test10._atomic_csv(root/'m4_validity_audit.csv', audit)
    base.capacity.test10._atomic_json(root/'feature_diagnostic.json', features['diagnostics'])
    print('준비만 완료. M1·수정 원형 M4(λ=.25) 재사용; 새 학습 M2·M5 각 1개, seed43.', flush=True)
    print('v3: N/V로 ID 표현을 조절. 초기 N/V 행렬은 0; 실제 강도·규제는 아래 settings 참조.', flush=True)
    print(json.dumps(dict(settings=settings, features=features['diagnostics'],
        max_epochs=cfg.epochs, final_test=False, holdout=False), ensure_ascii=False, indent=2))
    return cfg, prep, audit


def _build(prep, cfg):
    base.v3.set_seed(43)
    d, settings = prep['data'], prep['feature_settings']
    return NVModulatedLightGCN(n_users=d['n_users'], n_items=d['n_items'],
        features=prep['features'], adj=d['adj'], id_dim=cfg.id_dim,
        n_layers=cfg.n_layers, pref_reg=cfg.pref_reg, shared_l2=settings['shared_l2'],
        eta_n=settings['eta_n'], eta_v=settings['eta_v']).to(base.v3.DEVICE)


def diagnose(model, prep, cfg, spec, epoch):
    """Fixed TRAIN probe; no optimizer step, evaluation labels, or training RNG consumed."""
    d = prep['data']
    rng = np.random.default_rng(4301)
    ix = rng.choice(len(d['tr_u']), min(8192, len(d['tr_u'])), replace=False)
    negatives = base.components.m4_helpers.sample_uniform_negative_matrix(
        d['tr_u'][ix], d['tr_i'][ix], d['n_items'], d['pos_key'], rng, k=1).reshape(-1)
    u, p, n = [torch.as_tensor(a, device=base.v3.DEVICE, dtype=torch.long)
               for a in (d['tr_u'][ix], d['tr_i'][ix], negatives)]
    pos, neg = model._pair_components(u, p, n)
    rows = torch.nn.functional.softplus(sum(neg.values())-sum(pos.values()))
    if spec['weighted']:
        rows = rows*torch.as_tensor(prep['m4_weights'][ix], device=rows.device, dtype=rows.dtype)
    parameters = [layer.weight for layer in model.encoders.values()]
    gradients = torch.autograd.grad(rows.mean(), parameters)
    penalties = torch.autograd.grad(model.batch_l2(u, p, n), parameters)
    result = {**model.representation_diagnostics(), **model.training_gradient_diagnostics(),
              'probe_rows': len(ix), 'probe_bpr': float(rows.detach().mean())}
    for name, gradient, penalty in zip(model.encoders, gradients, penalties):
        result[name+'_bpr_gradient_norm'] = float(gradient.detach().double().norm())
        result[name+'_l2_gradient_norm'] = float(penalty.detach().double().norm())
    for name in pos:
        gap = (pos[name]-neg[name]).detach().double()
        result[name+'_pair_gap_mean_abs'] = float(gap.abs().mean())
        result[name+'_pair_gap_std'] = float(gap.std(unbiased=False))
    finite = all(np.isfinite(value) for value in result.values() if value is not None)
    suffix = '_bpr_gradient_norm' if epoch == 0 else '_modulation_mean_abs'
    result['nv_activity_ok'] = bool(finite and all(result[name+suffix] > 0 for name in model.encoders))
    print(f"[N/V probe ep{epoch}] active={result['nv_activity_ok']} | "
          f"N gap={result['n_pair_gap_mean_abs']:.3g} V gap={result['v_pair_gap_mean_abs']:.3g}", flush=True)
    return result


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
    diagnostics = pd.DataFrame([dict(model_id=a['model_id'], epoch=r['epoch'], **r['diagnostics'])
        for a in arms for r in a.get('training', {}).get('history', a['curve']) if 'diagnostics' in r])
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
        regularization=dict(id_coefficient=cfg.pref_reg, shared_nv_coefficient=prep['feature_settings']['shared_l2'],
            formula='pref_reg * sampled_ID_squared_sum / batch_size + shared_l2 * shared_NV_squared_sum'),
        representation=dict(formula='z_tilde = z * (1 + eta_N*tanh(A_N phi_N) + eta_V*tanh(A_V phi_V)); score = user_tilde dot item_tilde',
            initialization='four A matrices zero; ID matched random initialization, all jointly trained',
            shared_parameters=12*cfg.id_dim, prior_alpha_not_equivalent_to_eta=True),
        probe=dict(epochs=[0, 1, 5, 10, 'every evaluation'], source='8192 fixed TRAIN pairs at most, seed4301',
            interpretation='N/V/cross terms partition training pair score gaps; not Top-10 contributions or CLV attribution',
            abort='nonfinite diagnostic, zero initial BPR gradient, or zero N/V modulation at later probe'),
        feature_diagnostics=prep['features']['diagnostics'], features_sha256=prep['features_sha256'],
        selection=es.preflight(prior.lambda025.previous.prior.configure(cfg.out_dir))['selection'],
        primary='overall weighted hit@10 and weighted NDCG@10; M5 vs M4 and M1',
        guard='six overall Recall/NDCG metrics each >= .99 * M1',
        m2='both fixed historical q_N/q_V condition propagated ID representations; no q_C, no economic propagation',
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
            result = es._train(model, prep, cfg, spec, 43, store, root, diagnose=diagnose)
            arm = dict(**spec, seed=43, identity=ident, origin='trained_nv_modulated_id_joint', **result)
            base.capacity.test10._atomic_json(path, arm)
            store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
                best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arms.append(arm)
        paths = save(arms, cfg, prep)  # Preserve completed M2 readout if M5 is interrupted.
    return paths
