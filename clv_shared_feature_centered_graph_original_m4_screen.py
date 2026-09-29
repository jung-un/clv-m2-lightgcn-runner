"""One Dunnhumby M2+M3+M4 fit; all four prior models are readouts only."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import clv_linear_nv_original_m4_screen as original_m4
import clv_shared_feature_centered_graph_screen as joint
from clv_run_state import ProgressStore, RunIdentity, file_sha256
from clv_shared_feature_residual_m2 import SharedFeatureResidualLightGCN

VERSION = 'shared-feature-centered-graph-original-m4-seed43-development-v1'
MODEL_ID = 'm5_shared_feature_centered_graph_original_m4_es'
JOINT_REPORT_SHA = '72a247c6b17af8669e227f0f22252c592db98fefb10e03f5057a34b73eead45a'
M4_REPORT_SHA = '7bf4632dc2b1f53b324cac320ddf107856fa34d4f64d0cfd43bc1b34bacb9dce'
M4_LAMBDA = .5
PRIMARY = ('price_purchase_amount_weighted_hit@10', 'vndcg@10')


class JointWithOriginalM4(SharedFeatureResidualLightGCN):
    """Only the M5 arm accepts M4 row weights; standalone M2 stays plain BPR."""

    def weighted_bpr_loss(self, users, positives, negatives, weights):
        if self.disabled_axes or weights.shape != users.shape or not torch.isfinite(weights).all() \
                or not torch.all(weights > 0):
            raise ValueError('Valid positive M4 weights and all N/V axes required')
        pos, neg = self._pair_scores(users, positives, negatives)
        bpr = (weights * F.softplus(neg-pos)).mean()
        return bpr+self.batch_l2(users, positives, negatives), dict(
            bpr=float(bpr.detach()), p_correct=float((pos > neg).float().mean().detach()))


def _spec():
    return dict(model_id=MODEL_ID, role='M5_full_M2_plus_M3_plus_M4',
                kind='shared_feature_residual', graph='centered_value', weighted=True)


def _m4_source(path, combo, cfg):
    report = joint._report(path, M4_REPORT_SHA)
    expected = json.loads(json.dumps(asdict(original_m4.configure(str(Path(path).parent)))))
    actual = report.get('config', {})
    if (report.get('code_version') != original_m4.VERSION or report.get('final_test') is not False
            or report.get('holdout') is not False or report.get('selection') != joint.m2.selection(cfg)
            or any(actual.get(k) != v for k, v in expected.items() if k not in ('out_dir', 'reuse_dirs'))
            or [Path(p).name for p in actual.get('reuse_dirs', [])]
               != [Path(p).name for p in expected['reuse_dirs']]):
        raise ValueError('Original M4 lambda=.5 protocol differs; no new fit')
    matches = [a for a in report['arms'] if a['model_id'] == original_m4.M4 and a['seed'] == 43]
    if len(matches) != 1 or report['arms'][0]['metrics'] != combo['arms'][0]['metrics']:
        raise ValueError('M4/M1 source readouts differ from the exact joint report')
    arm = matches[0]
    selected = original_m4.es.replay(arm['curve'], original_m4.configure(cfg.out_dir))
    if (arm['selected_epoch'] != selected['selected_epoch']
            or arm['stopped_epoch'] != selected['stopped_epoch']
            or arm['metrics'] != selected['selected_record']['metrics']
            or set(arm['metrics']) != set(combo['arms'][0]['metrics'])):
        raise ValueError('M4 selection or metric definitions differ')
    checkpoint = Path(arm['checkpoint'])
    if not checkpoint.is_file() or file_sha256(checkpoint) != arm['checkpoint_sha256']:
        raise ValueError('Exact completed M4 checkpoint missing/changed')
    return arm, report


def prepare(m2_report, m3_report, joint_report, m4_report, out_dir):
    """All references, masks, graph and weights must pass before GPU training."""
    combo = joint._report(joint_report, JOINT_REPORT_SHA)
    if (combo.get('code_version') != joint.VERSION or combo.get('final_test') is not False
            or combo.get('holdout') is not False or combo.get('new_fit_count') != 1
            or [a['model_id'] for a in combo['arms']] !=
               ['m1', joint.m2.MODEL_ID, joint.m3.ARM_VALUE, joint.MODEL_ID]
            or combo.get('reference_reports', {}).get('m2_sha256') != joint.M2_REPORT_SHA
            or combo.get('reference_reports', {}).get('m3_sha256') != joint.M3_REPORT_SHA):
        raise ValueError('Exact prior M2+M3 development result required')
    cfg = joint.m2.configure(str(out_dir))
    m4_arm, m4_source = _m4_source(m4_report, combo, cfg)
    cfg, prep = joint.prepare(m2_report, m3_report, out_dir)
    expected_config = json.loads(json.dumps(asdict(cfg)))
    expected_config['out_dir'] = combo['config']['out_dir']
    expected_config['reuse_dirs'] = combo['config']['reuse_dirs']
    if (combo['config'] != expected_config
            or [Path(p).name for p in combo['config']['reuse_dirs']]
               != [Path(p).name for p in cfg.reuse_dirs]
            or combo['selection'] != joint.m2.selection(cfg)
            or combo['features_sha256'] != prep['features_sha256']
            or combo['graph']['sha256'] != prep['graph_sha256']
            or m4_arm['identity']['input_hash'] != prep['input_hash']):
        raise ValueError('Prior M2+M3/M4 inputs or training contract differ')
    old = combo['arms'][-1]
    selected = joint.m2.es.replay(old['curve'], cfg)
    if (old['selected_epoch'] != selected['selected_epoch']
            or old['metrics'] != selected['selected_record']['metrics']
            or old['identity']['input_hash'] != prep['input_hash']
            or old['identity']['graph_sha256'] != prep['graph_sha256']):
        raise ValueError('Prior joint checkpoint readout differs')
    checkpoint = Path(old['checkpoint'])
    if not checkpoint.is_file() or file_sha256(checkpoint) != old['checkpoint_sha256']:
        raise ValueError('Prior joint checkpoint missing/changed')
    audit, weights, diagnostic = original_m4.audit_weights(prep, original_m4.configure(cfg.out_dir))
    prior = m4_source['m4_diagnostics']
    if (not diagnostic['original_invalid_extra_absent']
            or len(weights) != len(prep['data']['tr_u'])
            or not np.isclose(weights.mean(), 1., atol=1e-8)
            or any(not np.isclose(diagnostic[k], prior[k], rtol=1e-5, atol=1e-7)
                   for k in ('train_mean_raw_weight', 'row_weight_cv', 'row_weight_min', 'row_weight_max'))):
        raise ValueError('M4 mask/weight audit differs from the exact lambda=.5 source')
    prep.update(anchors=[*prep['anchors'], dict(old, origin='exact_prior_M2_M3_joint_readout'),
                         dict(m4_arm, role='M4', origin='exact_prior_masked_original_M4_readout')],
                m4_weights=weights, m4_audit=audit, m4_diagnostics=diagnostic,
                m4_weight_sha256=hashlib.sha256(weights.astype(np.float32).tobytes()).hexdigest(),
                joint_report=str(joint_report), m4_report=str(m4_report), screen_config=asdict(cfg))
    print('학습 전 확인 완료: 기존 M1/M2/M3/M2+M3/M4는 readout만 재사용.', flush=True)
    print('새 학습 1개: M2+M3+원형 M4, lambda=.5, 무효행 추가 가중 0, N/V 유지.', flush=True)
    return cfg, prep


def _identity(prep, cfg):
    ident = joint.m2.base._identity(prep, joint.m2.es.strength_cfg(cfg, cfg.epochs), _spec(), 43)
    ident.update(version=VERSION, selection=joint.m2.selection(cfg),
        features_sha256=prep['features_sha256'], graph_sha256=prep['graph_sha256'],
        m4_weight_sha256=prep['m4_weight_sha256'], m4_lambda=M4_LAMBDA,
        joint_report_sha256=JOINT_REPORT_SHA, m4_report_sha256=M4_REPORT_SHA)
    ident['source_hashes'].update({name: file_sha256(Path(__file__).with_name(name))
        for name in (Path(__file__).name, 'clv_shared_feature_centered_graph_screen.py',
                     'clv_shared_feature_residual_m2.py', 'clv_linear_nv_original_m4_screen.py',
                     'lightgcn_clv_history_linear_nv_early_stop.py', 'clv_run_state.py')})
    return json.loads(json.dumps(ident))


def _build(prep, cfg):
    joint.m2.base.v3.set_seed(43)
    data = prep['data']
    return JointWithOriginalM4(n_users=data['n_users'], n_items=data['n_items'],
        adj=prep['weighted_adj'], features=prep['features'], id_dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        **{k: v for k, v in joint.m2.SETTINGS.items() if k != 'bandwidth'}).to(joint.m2.base.v3.DEVICE)


def run(cfg, prep):
    if cfg != joint.m2.configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    if (joint.m2.feature_hash(prep['features']) != prep['features_sha256']
            or hashlib.sha256(prep['m4_weights'].astype(np.float32).tobytes()).hexdigest()
               != prep['m4_weight_sha256']
            or file_sha256(prep['joint_report']) != JOINT_REPORT_SHA
            or file_sha256(prep['m4_report']) != M4_REPORT_SHA):
        raise ValueError('Prepared inputs or reference reports changed')
    ident = _identity(prep, cfg)
    key = joint.m2.base._digest(ident)
    root = Path(cfg.out_dir)/'arms'/key
    path = root/'result.json'
    model = _build(prep, cfg)
    if path.is_file():
        arm = json.loads(path.read_text())
        selected = joint.m2.es.replay(arm['curve'], cfg)
        if (arm['identity'] != ident or arm['metrics'] != selected['selected_record']['metrics']
                or file_sha256(arm['checkpoint']) != arm['checkpoint_sha256']):
            raise ValueError('Completed M5 checkpoint identity changed')
        print('[cached] 완료 M2+M3+M4 재사용; 추가 학습 없음', flush=True)
    else:
        store = ProgressStore(root/'progress', RunIdentity(VERSION, MODEL_ID, 43, key,
            'content:'+joint.m2.base._digest(ident['source_hashes']), prep['input_hash']))
        result = joint.m2.es._train(model, prep, cfg, _spec(), 43, store, root,
                                    diagnose=joint.m2.diagnose)
        arm = dict(**_spec(), seed=43, identity=ident,
                   origin='joint_training_from_random_initialization', **result)
        joint.m2.base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path),
            checkpoint_path=result['checkpoint'])
    anchors = prep['anchors']
    if any(set(a['metrics']) != set(arm['metrics']) for a in anchors):
        raise ValueError('M5 and reference metric definitions differ')
    absolute = pd.DataFrame([dict(model_id=a['model_id'], role=a.get('role', a['model_id']),
        origin=a['origin'], seed=43, selected_epoch=a['selected_epoch'],
        stopped_epoch=a['stopped_epoch'], **a['metrics']) for a in (*anchors, arm)])
    comparisons, same_epoch = [], []
    for ref in anchors:
        for metric, value in arm['metrics'].items():
            old = ref['metrics'][metric]
            comparisons.append(dict(model_id=MODEL_ID, reference=ref['model_id'], metric=metric,
                value=value, reference_value=old, delta=value-old,
                relative_change_pct=100*(value/old-1) if old else np.nan,
                model_epoch=arm['selected_epoch'], reference_epoch=ref['selected_epoch']))
        ref_curve = {r['epoch']: r['metrics'] for r in ref['curve']}
        for record in arm['curve']:
            epoch = record['epoch']
            if epoch not in ref_curve:
                continue
            for metric, value in record['metrics'].items():
                old = ref_curve[epoch][metric]
                same_epoch.append(dict(model_id=MODEL_ID, reference=ref['model_id'], epoch=epoch,
                    metric=metric, value=value, reference_value=old, delta=value-old,
                    relative_change_pct=100*(value/old-1) if old else np.nan))
    curve = pd.DataFrame([dict(model_id=a['model_id'], origin=a['origin'], epoch=r['epoch'],
        **r['metrics']) for a in (*anchors, arm) for r in a['curve']])
    diagnostics = pd.DataFrame([dict(epoch=r['epoch'], **r['diagnostics'])
        for r in arm['training']['history'] if 'diagnostics' in r])
    reading = dict(significance_claim=False, single_repeatedly_exposed_development_seed=True,
        six_accuracy_guard_vs_m1=all(arm['metrics'][k] >= .99*anchors[0]['metrics'][k]
                                     for k in joint.m2.base.ACCURACY),
        both_economic_at10_above_each_reference=all(
            arm['metrics'][k] > ref['metrics'][k] for k in PRIMARY for ref in anchors))
    paths, out = {}, Path(cfg.out_dir)/'reports'
    for name, frame in (('absolute', absolute), ('comparison', pd.DataFrame(comparisons)),
                        ('same_epoch_comparison', pd.DataFrame(same_epoch)), ('curve', curve),
                        ('diagnostics', diagnostics), ('m4_validity_audit', prep['m4_audit'])):
        paths[name] = str(out/f'{name}.csv')
        joint.m2.base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(out/'result.json')
    joint.m2.base.capacity.test10._atomic_json(Path(paths['json']), dict(code_version=VERSION,
        config=asdict(cfg), m4_lambda=M4_LAMBDA, selection=joint.m2.selection(cfg),
        config_note='Inherited rho/history fields and config positive_weight_lambda=.25 are unused; audited M4 lambda=.5 weights are applied',
        new_fit_count=1, new_arms=[_spec()], split='historical_development_days_684_690',
        task='new-to-user; train pairs excluded; MIN_ITEM_INTER=1; uniform K=1',
        representation='shared user N/V and item-price layer0; centered-value graph; one optimizer',
        loss='weighted BPR; valid raw row=1+.5*q_C*item amount percentile*user bin fit, invalid raw row=1, normalized to train mean 1',
        graph=dict(beta=prep['m3_beta'], audit=prep['graph_audit'], sha256=prep['graph_sha256']),
        feature_diagnostics=prep['features']['diagnostics'], features_sha256=prep['features_sha256'],
        m4_diagnostics=prep['m4_diagnostics'], m4_weight_sha256=prep['m4_weight_sha256'],
        reference_reports=dict(joint=prep['joint_report'], joint_sha256=JOINT_REPORT_SHA,
                               m4=prep['m4_report'], m4_sha256=M4_REPORT_SHA),
        m3_selection_note='retrospective replay of an older fixed-epoch curve',
        comparison_note='independently selected checkpoints; same-epoch table is exploratory',
        reading=reading, arms=[*anchors, arm], final_test=False, holdout=False, paths=paths,
        limits='Development-only one exposed seed; no significance, generalization, CLV attribution or synergy claim'))
    return paths
