"""One joint M2+M3 development fit; exact prior readouts, no baseline training."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import clv_shared_feature_residual_m2_screen as m2
import lightgcn_clv_m3_centered_value_graph as m3
from clv_run_state import ProgressStore, RunIdentity, file_sha256
from clv_shared_feature_residual_m2 import build_features, SharedFeatureResidualLightGCN

VERSION = 'shared-feature-centered-value-graph-seed43-development-v1'
MODEL_ID = 'm5_shared_feature_centered_value_graph_es'
M2_REPORT_SHA = 'dd79076ad54080fab73bca0e5b0a336e0581330beb15a260564bd5c9ae7e0b2e'
M3_REPORT_SHA = '2614b8d1711e032f54e94aae82286734f5dbd1ce12b36dcc5503e97603fd52e4'
M3_REVISION = 'b6c579fa869c3e383368ea473cb4351f9ff1beb4'
CORE = (*m2.base.ACCURACY, 'price_purchase_amount_weighted_hit@10', 'vndcg@10',
        'price_purchase_amount_weighted_hit@20', 'vndcg@20',
        'price_purchase_amount_weighted_hit@50', 'vndcg@50')
PRIMARY = ('price_purchase_amount_weighted_hit@10', 'vndcg@10')


def _report(path, digest):
    path = Path(path)
    if file_sha256(path) != digest:
        raise ValueError(f'Exact reference report required: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def _m3_arm(report, cfg):
    rows = [r for r in report['curve'] if r['seed'] == 43 and r['model_id'] == m3.ARM_VALUE]
    if [r['epoch'] for r in rows] != list(range(25, 301, 25)):
        raise ValueError('Complete chronological seed43 M3 curve required')
    metadata = {'model_id', 'arm', 'gamma', 'seed', 'epoch', 'loss'}
    curve = [dict(epoch=r['epoch'], metrics={k: v for k, v in r.items() if k not in metadata})
             for r in rows]
    selected = m2.es.replay(curve, cfg)
    return dict(model_id=m3.ARM_VALUE, role='M3', seed=43,
                origin='replayed_prior_fixed_epoch_m3_curve',
                selected_epoch=selected['selected_epoch'], stopped_epoch=selected['stopped_epoch'],
                metrics=selected['selected_record']['metrics'], curve=selected['curve'])


def prepare(m2_report_path, m3_report_path, out_dir):
    """Check both exact prior runs before any fit; replay M3 selection retrospectively."""
    cfg = m2.configure(str(out_dir))
    old_m2 = _report(m2_report_path, M2_REPORT_SHA)
    old_m3 = _report(m3_report_path, M3_REPORT_SHA)
    if (old_m2['code_version'] != m2.VERSION or old_m2['final_test'] or old_m2['holdout']
            or old_m2['selection'] != m2.selection(cfg)
            or old_m3['code_version'] != m3.CODE_VERSION
            or old_m3['source_revision'] != M3_REVISION):
        raise ValueError('Reference run/version/selection mismatch')
    expected = {k: getattr(cfg, k) for k in ('epochs', 'id_dim', 'n_layers', 'batch_size',
                'lr', 'pref_reg', 'negative_count')}
    if any(old_m2['config'][k] != v or old_m3['config'][k] != v for k, v in expected.items()):
        raise ValueError('Reference training protocols differ')
    arms = old_m2['arms']
    if [a['model_id'] for a in arms] != ['m1', m2.MODEL_ID]:
        raise ValueError('Exact M1 and shared-feature M2 readouts required')
    for arm in arms:
        m2.source.check_selected(arm, cfg)
    m3_arm = _m3_arm(old_m3, cfg)
    if set(m3_arm['metrics']) != set(arms[0]['metrics']):
        raise ValueError('M3 metric definitions differ')
    prep = m2.base._prepare(m2.es.strength_cfg(cfg))
    data, axes = prep['data'], prep['axes']
    if (data['splits'].keys() != {'test'} or prep['base_cfg']['EVAL_HOLDOUT']
            or prep['base_cfg']['MIN_ITEM_INTER'] != 1 or float(data['train'].t.max()) != 683.):
        raise ValueError('Development-only new-item protocol required')
    features = build_features(data['train'], n_users=data['n_users'], n_items=data['n_items'],
        q_n=axes['q_n'], q_v=axes['q_v'], n_valid=axes['activity_valid'],
        v_valid=axes['value_valid'], bandwidth=m2.SETTINGS['bandwidth'])
    features['keys'] = np.asarray(data['pos_key'], np.int64).copy()
    if (prep['input_hash'] != arms[1]['identity']['input_hash']
            or m2.feature_hash(features) != old_m2['features_sha256']):
        raise ValueError('M2 train input/features differ from exact reference')
    old_m3_cfg = m3.CenteredGraphConfig(**old_m3['config'])
    if (m3._config_hash(old_m3_cfg, prep['input_hash']) != '244783a20cc5'
            or old_m3_cfg.target_cv != .20):
        raise ValueError('M3 input/config identity differs from exact reference')
    mismatches = []
    m1_by_epoch = {r['epoch']: r['metrics'] for r in arms[0]['curve']}
    for row in old_m3['curve']:
        if row['seed'] != 43 or row['model_id'] != m3.M1_MODEL_ID or row['epoch'] not in m1_by_epoch:
            continue
        for metric, value in m1_by_epoch[row['epoch']].items():
            if not np.isclose(value, row[metric], rtol=1e-8, atol=1e-10):
                mismatches.append((row['epoch'], metric, value, row[metric]))
    if any(metric in CORE for _, metric, _, _ in mismatches):
        raise ValueError('M1 core readouts differ across reference runs')
    signals = m3.centered_edge_signals(data['train'], data['n_users'], data['n_items'])
    if not np.array_equal(signals['edge_users']*data['n_items']+signals['edge_items'], data['pos_key']):
        raise ValueError('M3 weighted edges differ from binary graph pairs')
    valid = np.asarray(prep['clv_valid'], bool)
    qv = np.where(valid, prep['q_v'], 0.)
    qn = np.where(valid, prep['q_n'], 0.)
    beta = old_m3['betas'][m3.ARM_VALUE]
    weights = m3.edge_weights(signals, qv, qn, gamma=0., beta=beta)
    audit = m3.audit_weights(weights, signals, prep)
    m3.check_gates(audit, old_m3_cfg, m3.ARM_VALUE)
    audit_mismatches = [(key, audit[key], prior)
        for key, prior in old_m3['edge_audits'][m3.ARM_VALUE].items()
        if not np.isclose(audit[key], prior, rtol=1e-4, atol=1e-5)]
    # The two Spearman values differ from the prior run; keep the discrepancy
    # visible, while requiring graph-strength/margin/range statistics to match.
    if any(key not in {'item_weight_vs_price', 'item_weight_vs_popularity'}
           for key, _, _ in audit_mismatches):
        raise ValueError(f'M3 graph statistics differ from prior run: {audit_mismatches}')
    prep.update(features=features, features_sha256=m2.feature_hash(features),
        feature_settings=m2.SETTINGS, graph_audit=audit, m3_beta=beta,
        graph_sha256=hashlib.sha256(weights.astype(np.float32).tobytes()).hexdigest(),
        weighted_adj=m2.base.v3.build_adj(signals['edge_users'], signals['edge_items'],
            weights.astype(np.float32), data['n_users'], data['n_items']),
        anchors=[dict(arms[0], origin='exact_M2_report_M1_readout'),
                 dict(arms[1], origin='exact_M2_report_M2_readout'), m3_arm],
        m1_mismatch=mismatches, m3_audit_mismatch=audit_mismatches,
        screen_config=asdict(cfg),
        m2_report=str(m2_report_path), m3_report=str(m3_report_path))
    print(f'학습 전 확인 완료: M1 공통지표 일치; 전체지표 불일치 {len(mismatches)}개 보존.', flush=True)
    print(f'M3 가중치 CV/여백/최솟값/최댓값 일치; 순위상관 차이 {len(audit_mismatches)}개 기록.', flush=True)
    print(f'M3 사후 조기종료 재판독: 선택 {m3_arm["selected_epoch"]}, 중단 {m3_arm["stopped_epoch"]}.', flush=True)
    print('새 학습은 M2+M3 한 개뿐. N/V 모두 유지, M4/H&M/test/holdout 없음.', flush=True)
    return cfg, prep


def _spec():
    return dict(model_id=MODEL_ID, role='M5_partial_M2_plus_M3', kind='shared_feature_residual',
                graph='centered_value', weighted=False)


def _identity(prep, cfg):
    ident = m2.base._identity(prep, m2.es.strength_cfg(cfg, cfg.epochs), _spec(), 43)
    ident.update(version=VERSION, selection=m2.selection(cfg),
        features_sha256=prep['features_sha256'], graph_sha256=prep['graph_sha256'],
        m2_report_sha256=M2_REPORT_SHA, m3_report_sha256=M3_REPORT_SHA)
    ident['source_hashes'].update({name: file_sha256(Path(__file__).with_name(name))
        for name in (Path(__file__).name, 'clv_shared_feature_residual_m2.py',
                     'clv_shared_feature_residual_m2_screen.py',
                     'lightgcn_clv_m3_centered_value_graph.py',
                     'lightgcn_clv_history_linear_nv_early_stop.py', 'clv_run_state.py')})
    return json.loads(json.dumps(ident))


def _build(prep, cfg):
    m2.base.v3.set_seed(43)
    data = prep['data']
    return SharedFeatureResidualLightGCN(n_users=data['n_users'], n_items=data['n_items'],
        adj=prep['weighted_adj'], features=prep['features'], id_dim=cfg.id_dim,
        n_layers=cfg.n_layers, **{k: v for k, v in m2.SETTINGS.items() if k != 'bandwidth'}).to(m2.base.v3.DEVICE)


def run(cfg, prep):
    if cfg != m2.configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    if (m2.feature_hash(prep['features']) != prep['features_sha256']
            or file_sha256(prep['m2_report']) != M2_REPORT_SHA
            or file_sha256(prep['m3_report']) != M3_REPORT_SHA):
        raise ValueError('Prepared inputs or reference reports changed')
    ident = _identity(prep, cfg)
    key = m2.base._digest(ident)
    root = Path(cfg.out_dir)/'arms'/key
    path = root/'result.json'
    model = _build(prep, cfg)
    if path.is_file():
        arm = json.loads(path.read_text())
        if arm['identity'] != ident or file_sha256(arm['checkpoint']) != arm['checkpoint_sha256']:
            raise ValueError('Completed joint checkpoint identity changed')
        m2.source.check_selected(arm, cfg)
        print('[cached] 완료 M2+M3 재사용; 추가 학습 없음', flush=True)
    else:
        store = ProgressStore(root/'progress', RunIdentity(VERSION, MODEL_ID, 43, key,
            'content:'+m2.base._digest(ident['source_hashes']), prep['input_hash']))
        result = m2.es._train(model, prep, cfg, _spec(), 43, store, root, diagnose=m2.diagnose)
        arm = dict(**_spec(), seed=43, identity=ident, origin='joint_training_from_random_initialization', **result)
        m2.base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path), checkpoint_path=result['checkpoint'])
    anchors = prep['anchors']
    if set(arm['metrics']) != set(anchors[0]['metrics']):
        raise ValueError('Joint metric definitions differ from references')
    all_arms = anchors+[arm]
    absolute = pd.DataFrame([dict(model_id=a['model_id'], role=a.get('role', a['model_id']),
        origin=a['origin'], seed=43, selected_epoch=a['selected_epoch'],
        stopped_epoch=a['stopped_epoch'], **a['metrics']) for a in all_arms])
    comparisons = []
    same_epoch = []
    for reference in anchors:
        for metric, value in arm['metrics'].items():
            old = reference['metrics'][metric]
            comparisons.append(dict(model_id=MODEL_ID, reference=reference['model_id'],
                metric=metric, value=value, reference_value=old, delta=value-old,
                relative_change_pct=100*(value/old-1) if old else np.nan,
                model_epoch=arm['selected_epoch'], reference_epoch=reference['selected_epoch']))
        reference_curve = {r['epoch']: r['metrics'] for r in reference['curve']}
        for record in arm['curve']:
            epoch = record['epoch']
            if epoch not in reference_curve:
                continue
            for metric, value in record['metrics'].items():
                old = reference_curve[epoch][metric]
                same_epoch.append(dict(model_id=MODEL_ID, reference=reference['model_id'],
                    epoch=epoch, metric=metric, value=value, reference_value=old,
                    delta=value-old, relative_change_pct=100*(value/old-1) if old else np.nan))
    curve = pd.DataFrame([dict(model_id=a['model_id'], origin=a['origin'], epoch=r['epoch'],
        **r['metrics']) for a in all_arms for r in a['curve']])
    diagnostics = pd.DataFrame([dict(epoch=r['epoch'], **r['diagnostics'])
        for r in arm['training']['history'] if 'diagnostics' in r])
    reading = dict(significance_claim=False, single_repeatedly_exposed_development_seed=True,
        six_accuracy_guard_vs_m1=all(arm['metrics'][k] >= .99*anchors[0]['metrics'][k]
                                  for k in m2.base.ACCURACY),
        both_economic_at10_above_m3=all(arm['metrics'][k] > anchors[2]['metrics'][k]
                                      for k in PRIMARY),
        both_economic_at10_above_m1=all(arm['metrics'][k] > anchors[0]['metrics'][k]
                                      for k in PRIMARY))
    paths, out = {}, Path(cfg.out_dir)/'reports'
    for name, frame in (('absolute', absolute), ('comparison', pd.DataFrame(comparisons)),
                        ('same_epoch_comparison', pd.DataFrame(same_epoch)),
                        ('curve', curve), ('diagnostics', diagnostics)):
        paths[name] = str(out/f'{name}.csv')
        m2.base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(out/'result.json')
    m2.base.capacity.test10._atomic_json(Path(paths['json']), dict(code_version=VERSION,
        config=asdict(cfg), selection=m2.selection(cfg), new_fit_count=1,
        new_arms=[_spec()], split='historical_development_days_684_690',
        task='new-to-user; train pairs excluded; MIN_ITEM_INTER=1; uniform K=1; plain BPR',
        representation='shared user N/V and item-price layer0 in centered-value weighted LightGCN graph; one optimizer',
        graph=dict(beta=prep['m3_beta'], audit=prep['graph_audit'], sha256=prep['graph_sha256']),
        feature_diagnostics=prep['features']['diagnostics'], features_sha256=prep['features_sha256'],
        reference_reports=dict(m2=prep['m2_report'], m2_sha256=M2_REPORT_SHA,
                               m3=prep['m3_report'], m3_sha256=M3_REPORT_SHA),
        m3_selection_note='retrospective replay of an older fixed-epoch curve; not original M3 preregistration',
        same_epoch_note='observed common epochs only; no interpolation or further baseline fitting',
        m1_cross_run_mismatches=prep['m1_mismatch'],
        m3_audit_cross_run_mismatches=prep['m3_audit_mismatch'],
        reading=reading, arms=all_arms, final_test=False, holdout=False, paths=paths,
        limits='Development-only one seed; no significance, generalization, CLV attribution or M5 synergy claim'))
    return paths
