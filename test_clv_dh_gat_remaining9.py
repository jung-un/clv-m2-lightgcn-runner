from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

import clv_dh_gat_remaining9 as s
from test_clv_dh_gat_final8_seed49 import prepared as gat_fixture
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as reference_fixture


def make_row(p, label, seed):
    ident = asdict(s.identity(p, label, seed)) if seed != 49 else {
        'source_revision': p['revision'], 'input_hash': p['input_hash']}
    return {'identity': ident, 'model': label, 'model_id': f'gat_{label.lower()}_lastweek',
        'seed': seed, 'epochs': 300,
        'metrics': {m: (j+1) + seed / 100 + s.ORDER.index(label) / 10
                    for j, m in enumerate(p['references']['metrics'])},
        'training_history': [{'epoch': i} for i in range(1, 301)],
        'diagnostics': {'graph': 'centered_nv' if s.ARMS[label][1] else 'binary',
                        'row_weighted': s.ARMS[label][2], 'feature_sha256': p['feature_sha256']}}


@pytest.fixture
def prepared(gat_fixture):
    p = gat_fixture
    p['cfg'] = s.configure(out_dir=str(p['run_dir']))
    metrics = list(s.CORE) + [f'metric_{i}' for i in range(130)]
    p['references']['metrics'] = metrics
    p['previous'] = {'rows': {m: make_row(p, m, 49) for m in s.ORDER},
                     'blobs': {'protocol.json': b'{}'}, 'audit': {}, 'metrics': metrics}
    return p


def test_config_and_seed49_never_trains(prepared, monkeypatch):
    s.validate_config(prepared['cfg'])
    assert s.SEEDS == tuple(range(42, 52))
    assert s.NEW_SEEDS == (42, 43, 44, 45, 46, 47, 48, 50, 51)
    monkeypatch.setattr(s.original, 'build_model', lambda *a: pytest.fail('seed49 model created'))
    with pytest.raises(ValueError, match='재사용 전용'):
        s.run_arm(prepared, 'M1', 49)
    for change in ({'seeds': (42,)}, {'epochs': 100}, {'lr': .01}, {'feature_dropout': .1},
                   {'input_days': 704}, {'batch_size': 16384}):
        with pytest.raises(ValueError):
            s.validate_config(replace(prepared['cfg'], **change))
    for path in (prepared['run_dir'], prepared['run_dir'] / 'inside', prepared['run_dir'].parent):
        with pytest.raises(ValueError, match='겹치지'):
            s.validate_config(replace(prepared['cfg'], seed49_reference=str(path)))


def test_partial_and_complete_72_runs_skip_and_export(prepared, monkeypatch):
    fits, evaluations = [], []
    def train(model, p, cfg, mid, seed, store, row_weights=None, *, negative_sampler=None):
        assert negative_sampler is s.original.sample_uniform_negative_matrix
        assert seed in s.NEW_SEEDS and cfg.epochs == 300
        fits.append((mid, seed))
        return [{'epoch': i} for i in range(1, 301)]
    def evaluate(model, p):
        evaluations.append(1)
        return dict.fromkeys(p['references']['metrics'], 1.0)
    monkeypatch.setattr(s.original.execution, 'train', train)
    monkeypatch.setattr(s.base.fixed, '_evaluate', evaluate)
    initial = s.report(prepared)
    assert len(initial['absolute']) == 8 and not initial['complete']
    assert 'summary' not in initial and 'core_mean_std' not in initial['paths']
    with pytest.raises(RuntimeError, match='미완료'):
        s.export_zip(prepared, prepared['run_dir'] / 'r.zip')
    partial = s.export_zip(prepared, prepared['run_dir'] / 'r.zip', allow_partial=True)
    assert partial.endswith('_partial.zip')
    actual = s.run(prepared)
    assert actual['complete'] and len(actual['absolute']) == 80
    assert set(actual['absolute'].seed) == set(s.SEEDS)
    assert len(actual['summary']) == 8 * 142
    assert len(actual['comparison_vs_m1']) == 7 * 142
    assert len(actual['paired_vs_m1']) == 7 * 142 * 10
    assert len(actual['core_mean_std']) == 12
    assert (actual['absolute'].source == 'reused_seed49').sum() == 8
    s.run(prepared)
    assert len(fits) == len(set(fits)) == len(evaluations) == 72
    with ZipFile(s.export_zip(prepared, prepared['run_dir'] / 'complete.zip')) as z:
        assert len([n for n in z.namelist() if n.startswith('arms/')]) == 72
        assert 'reused/seed49/protocol.json' in z.namelist()
        assert 'core_mean_std.csv' in z.namelist()
        assert not any(n.endswith('.pt') for n in z.namelist())


def test_balanced_aggregation_independent_numpy_and_invalid_rows(prepared):
    metrics = prepared['references']['metrics']
    a = pd.DataFrame([{'model': m, 'seed': seed, **make_row(prepared, m, seed)['metrics']}
                      for seed in s.SEEDS for m in s.ORDER])
    tables = s.aggregate(a.sample(frac=1, random_state=20), metrics)
    summary = tables['summary'].set_index(['model', 'metric'])
    for m in s.ORDER:
        for metric in metrics:
            vals = a[a.model == m][metric].to_numpy()
            assert summary.loc[(m, metric), 'mean'] == pytest.approx(sum(vals)/10)
            sd = (sum((v - sum(vals)/10)**2 for v in vals)/9)**.5
            assert summary.loc[(m, metric), 'std'] == pytest.approx(sd)
    comp = tables['comparison_vs_m1'].iloc[0]
    assert comp['paired_delta_mean'] == pytest.approx(.1)
    x = a[a.model == 'M2'][metrics[0]].mean()
    y = a[a.model == 'M1'][metrics[0]].mean()
    assert comp['relative_change_pct'] == pytest.approx(100*(x/y-1))
    assert s.aggregate(a.iloc[:-1], metrics) == {}
    for bad in (pd.concat([a, a.iloc[:1]]), a.assign(seed=99), a.assign(**{metrics[0]: np.nan})):
        with pytest.raises(RuntimeError):
            s.aggregate(bad, metrics)


@pytest.mark.parametrize('label', s.ORDER)
def test_new_seed_training_identical_to_seed49_path_and_resume(prepared, tmp_path, label):
    cfg = replace(prepared['cfg'], epochs=2, batch_size=3)
    ident = s.identity(prepared, label, 50)
    def fresh():
        s.base.v3.set_seed(50)
        model, weights = s.original.build_model(prepared, label)
        s.base.v3.set_seed(50)
        return model, weights
    full, weights = fresh()
    full_store = s.original.SafeProgressStore(tmp_path / 'full', ident)
    history = s.original.execution.train(full, prepared, cfg, ident.model_id, 50, full_store, row_weights=weights, negative_sampler=s.original.sample_uniform_negative_matrix)
    part, _ = fresh()
    store = s.original.SafeProgressStore(tmp_path / 'resume', ident)
    s.original.execution.train(part, prepared, replace(cfg, epochs=1), ident.model_id, 50, store, row_weights=weights, negative_sampler=s.original.sample_uniform_negative_matrix)
    resumed, _ = fresh()
    actual = s.original.execution.train(resumed, prepared, cfg, ident.model_id, 50, store, row_weights=weights, negative_sampler=s.original.sample_uniform_negative_matrix)
    assert actual == history
    a, b = (torch.load(p.latest_checkpoint, weights_only=False) for p in (full_store, store))
    for key in full.state_dict():
        torch.testing.assert_close(full.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)
    assert a['numpy_rng_state'] == b['numpy_rng_state']
    torch.testing.assert_close(a['torch_rng_state'], b['torch_rng_state'], rtol=0, atol=0)
    for key, value in a['optimizer_state']['state'].items():
        for name in value:
            torch.testing.assert_close(value[name], b['optimizer_state']['state'][key][name], rtol=0, atol=0)


def test_saved_identity_and_early_evaluation_guard(prepared, monkeypatch):
    p = s.arm_path(prepared, 'M1', 42)
    p.parent.mkdir(parents=True)
    row = make_row(prepared, 'M1', 42)
    row['identity']['seed'] = 43
    p.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match='신원'):
        s.completed(prepared, 'M1', 42)
    monkeypatch.setattr(s.original.execution, 'train', lambda *a, **kw: [{'epoch': 299}])
    monkeypatch.setattr(s.base.fixed, '_evaluate', lambda *a: pytest.fail('early test evaluation'))
    with pytest.raises(RuntimeError, match='300epoch'):
        s.run_arm(prepared, 'M2', 42)


def synthetic_source(p, monkeypatch):
    protocol = {'source_revision': p['revision'], 'input_hash': p['input_hash'],
                'feature_sha256': p['feature_sha256']}
    rows = p['previous']['rows']
    absolute = pd.DataFrame([{'model': r['model'], 'model_id': r['model_id'], 'seed': 49, **r['metrics']}
                            for r in rows.values()])
    blobs = {'absolute.csv': absolute.to_csv(index=False).encode(),
             'protocol.json': json.dumps(protocol).encode(),
             'result.json': json.dumps({'protocol': protocol, 'reading': {'complete': True}}).encode(),
             'diagnostics.csv': b'x\n', 'comparison.csv': b'x\n'}
    blobs.update({f"arms/{r['model_id']}_s49.json": json.dumps(r).encode() for r in rows.values()})
    monkeypatch.setattr(s, 'OLD_SHA', {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()})
    return blobs


def test_seed49_zip_root_and_reports_and_corruption(prepared, monkeypatch, tmp_path):
    blobs = synthetic_source(prepared, monkeypatch)
    parent = tmp_path / 'old'
    root = parent / 'hash'
    for n, b in blobs.items():
        p = root / n if n.startswith('arms/') or n == 'protocol.json' else root / 'reports' / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    zpath = tmp_path / 'old.zip'
    with ZipFile(zpath, 'w') as z:
        for n, b in blobs.items():
            z.writestr(n, b)
    for path in (parent, root, root / 'reports', zpath):
        previous = s.load_seed49(path)
        assert len(previous['rows']) == 8 and len(previous['metrics']) == 142
    (root / 'arms/gat_m1_lastweek_s49.json').write_bytes(b'broken')
    with pytest.raises(RuntimeError, match='재학습하지'):
        s.load_seed49(parent)
    with pytest.raises(RuntimeError, match='재학습하지'):
        s.load_seed49(tmp_path / 'missing.zip')


def test_prepare_once_inputs_and_source_consistency(prepared, monkeypatch):
    oldcfg = s.original.configure()
    protocol = {'config': json.loads(json.dumps(asdict(oldcfg))), 'source_revision': 'test',
        'code_sha256': {}, 'input_hash': prepared['input_hash'], 'data_stats': {},
        'split': 'train DAY 1-704; test DAY 705-711', 'feature_sha256': prepared['feature_sha256'],
        'm3_audit': {}, 'm4_weight_audit': {}, 'dataset_initialization': {},
        'fixed_intervention_settings': {}, 'models': {}, 'backbone': 'GAT', 'propagation': 'same',
        'adjacency': 'same', 'regularization': 'same', 'm2_adaptation': 'same', 'initialization': 'same'}
    previous = {**prepared['previous'], 'protocol': protocol}
    calls = []
    def prepare(cfg):
        calls.append(cfg)
        return {**prepared, 'protocol': copy.deepcopy(protocol)}
    monkeypatch.setattr(s, 'load_seed49', lambda *a: previous)
    monkeypatch.setattr(s, 'check_source_compatibility', lambda *a: {})
    monkeypatch.setattr(s.original, 'prepare', prepare)
    monkeypatch.setattr(s.original.execution, 'train', lambda *a, **kw: pytest.fail('prepare trained'))
    monkeypatch.setattr(s.base.fixed, '_evaluate', lambda *a: pytest.fail('prepare evaluated'))
    actual = s.prepare(prepared['cfg'])
    assert len(calls) == 1 and calls[0].seeds == (49,)
    assert actual['features'] is prepared['features']
    assert actual['observed_adj'] is prepared['observed_adj']
    assert actual['observed_weights'] is prepared['observed_weights']
    assert actual['protocol']['new_fit_count'] == 72
    assert actual['protocol']['reused_gat_results'] == 8
    again = s.prepare(prepared['cfg'])
    assert again['config_hash'] == actual['config_hash']
    def changed(cfg):
        p = prepare(cfg)
        p['protocol']['feature_sha256'] = 'changed'
        return p
    monkeypatch.setattr(s.original, 'prepare', changed)
    with pytest.raises(RuntimeError, match='feature_sha256'):
        s.prepare(prepared['cfg'])


def test_audit_platform_roundoff_does_not_hide_changed_weights():
    old = {"m3_audit": {"coefficient_of_variation": .2, "item_weight_vs_price": -.005},
           "m4_weight_audit": {"sha256": "exact", "row_weight_cv": .13}}
    new = copy.deepcopy(old)
    new["m3_audit"]["coefficient_of_variation"] += 1e-16
    new["m3_audit"]["item_weight_vs_price"] = -.007
    assert len(s.compare_audits(old, new)["differences"]) == 2
    new["m4_weight_audit"]["sha256"] = "changed"
    with pytest.raises(RuntimeError): s.compare_audits(old, new)
    new = copy.deepcopy(old)
    new["m3_audit"]["coefficient_of_variation"] = .21
    with pytest.raises(RuntimeError): s.compare_audits(old, new)
