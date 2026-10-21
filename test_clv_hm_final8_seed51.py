from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

import clv_hm_final8_seed51 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared


@pytest.fixture
def hm(prepared, tmp_path):
    p = prepared
    p['cfg'] = s.configure(out_dir=str(tmp_path / 'new'))
    p['observed_weights'] = np.asarray(p['observed_weights'], np.float32)
    p['protocol'] = {'m4_weight_audit': {'sha256': hashlib.sha256(p['observed_weights'].tobytes()).hexdigest()},
                     'm3_audit': {'cv': .2}}
    p['data'].update(pos_key=p['features']['keys'], tr_u=np.array([0, 0, 1, 1, 2, 2]),
                     tr_i=np.array([0, 1, 1, 2, 2, 3]))
    p.update(q_n=np.ones(3, np.float32)*.5, q_v=np.ones(3, np.float32)*.5,
             q_c=np.ones(3, np.float32)*.5, clv_valid=np.ones(3, bool),
             item_amount_percentile=np.ones(5, np.float32)*.5, item_economic_valid=np.ones(5, bool))
    for n in ('protocol.json', 'reference_audit.json'):
        s.base.io._atomic_json(p['run_dir'] / n, {})
    torch.set_num_threads(1)
    return p


def test_routes_and_frozen_config(hm):
    assert list(s.ORDER) == ['M1', 'M2', 'M3', 'M4', 'M5_A', 'M5_B', 'M5_C', 'M5_D']
    for label, bits in s.ARMS.items():
        adj, w = s.arm_inputs(hm, label)
        assert adj is (hm['observed_adj'] if bits[1] else hm['data']['adj'])
        assert (w is not None) == bits[2]
        model, _ = s.build_model(hm, label)
        assert isinstance(model, s.base.m2.DatasetAdaptiveOrthogonalNVLightGCN) == bits[0]
    for change in ({'seeds': (50,)}, {'epochs': 100}, {'batch_size': 8192}, {'input_days': 704},
                   {'lr': .001}, {'shared_l2': 0}, {'negative_count': 5}):
        with pytest.raises(ValueError):
            s.validate_config(replace(hm['cfg'], **change))
    with pytest.raises(ValueError, match='겹칠'):
        s.validate_config(replace(hm['cfg'], out_dir=hm['cfg'].m2_reference + '/nested'))


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('label', s.ORDER)
def test_optimized_train_exactly_matches_original_and_resume(hm, label, tmp_path, monkeypatch):
    cfg = replace(hm['cfg'], epochs=3, batch_size=4)
    ident = s.identity(hm, label)
    original, w = s.build_model(hm, label)
    fast, _ = s.build_model(hm, label)
    old_store = s.schema.ProgressStore(tmp_path/'old', ident)
    fast_store = s.ProgressStore(tmp_path/'fast', ident)
    original_history = s.base.fixed._train(original, hm, cfg, ident.model_id, 51, old_store, row_weights=w)
    def train(model, store):
        return s.execution.train(model, hm, cfg, ident.model_id, 51, store, row_weights=w,
            negative_sampler=s.sample_uniform_negative_matrix, loss_function=s.loss_with_tensor_logs)
    history = train(fast, fast_store)
    assert history == original_history
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, fast.state_dict()[k], rtol=0, atol=0)
    old = torch.load(old_store.latest_checkpoint, weights_only=False)
    new = torch.load(fast_store.latest_checkpoint, weights_only=False)
    for k in ('model_state', 'optimizer_state', 'numpy_rng_state', 'torch_rng_state', 'history'):
        assert_nested_equal(old[k], new[k])
    split_store = s.ProgressStore(tmp_path/'split', ident)
    save = split_store.save_epoch
    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if kwargs['epoch'] == 1: raise RuntimeError('simulated interruption')
    monkeypatch.setattr(split_store, 'save_epoch', interrupt)
    resumed, _ = s.build_model(hm, label)
    with pytest.raises(RuntimeError, match='simulated'):
        train(resumed, split_store)
    monkeypatch.setattr(split_store, 'save_epoch', save)
    resumed, _ = s.build_model(hm, label)
    assert train(resumed, split_store) == history
    split = torch.load(split_store.latest_checkpoint, weights_only=False)
    for k in ('model_state', 'optimizer_state', 'numpy_rng_state', 'torch_rng_state', 'history'):
        assert_nested_equal(new[k], split[k])


def test_sampling_exact_output_and_rng(hm):
    u, i, keys = (hm['data'][k] for k in ('tr_u', 'tr_i', 'pos_key'))
    for seed in (49, 50, 51):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        for _ in range(20):
            x = s.base.fixed.m4_helpers.sample_uniform_negative_matrix(u, i, 5, keys, a, k=1)
            y = s.sample_uniform_negative_matrix(u, i, 5, keys, b, k=1)
            np.testing.assert_array_equal(x, y)
            assert a.bit_generator.state == b.bit_generator.state


def test_eight_fits_completed_skip_partial_export(hm, monkeypatch):
    fits, evaluations = [], []
    def train(model, p, cfg, mid, seed, store, row_weights=None, **kwargs):
        assert cfg.seeds == (51,) and cfg.epochs == 300 and cfg.batch_size == 131072
        fits.append(mid)
        return [{'epoch': i} for i in range(1, 301)]
    def evaluate(*args):
        evaluations.append(True)
        return dict.fromkeys(hm['references']['metrics'], 1.01)
    monkeypatch.setattr(s.execution, 'train', train)
    monkeypatch.setattr(s.base.fixed, '_evaluate', evaluate)
    assert not s.report(hm)['complete']
    with pytest.raises(RuntimeError, match='未|미완료'):
        s.export_zip(hm, hm['run_dir']/'bad.zip')
    s.run_arm(hm, 'M1')
    assert len(s.report(hm)['absolute']) == 1
    result = s.run(hm)
    s.run(hm)
    assert fits == list(s.IDS[label] for label in s.ORDER) and len(evaluations) == 8
    assert result['complete'] and set(result['absolute'].seed) == {51}
    assert len(result['comparison']) == 8*7*142
    with ZipFile(s.export_zip(hm, hm['run_dir']/'result.zip')) as z:
        assert len([n for n in z.namelist() if n.startswith('arms/')]) == 8
        assert not any(n.endswith('.pt') for n in z.namelist())


def test_early_evaluation_bad_result_and_weights_block(hm, monkeypatch):
    monkeypatch.setattr(s.execution, 'train', lambda *a, **kw: [{'epoch': 299}])
    monkeypatch.setattr(s.base.fixed, '_evaluate', lambda *a: pytest.fail('early test'))
    with pytest.raises(RuntimeError, match='300epoch'): s.run_arm(hm, 'M1')
    s.base.io._atomic_json(s.arm_path(hm, 'M2'), {'seed': 50})
    with pytest.raises(RuntimeError, match='신원'): s.run_arm(hm, 'M2')
    hm['observed_weights'] = None
    with pytest.raises(RuntimeError, match='가중치 누락'): s.run_arm(hm, 'M4')


def test_corrupt_checkpoint_and_drive_failure_preserve_previous(hm, tmp_path, monkeypatch):
    store = s.ProgressStore(tmp_path/'progress', s.identity(hm, 'M1'))
    store.latest_checkpoint.parent.mkdir(parents=True)
    store.latest_checkpoint.write_bytes(b'previous')
    with pytest.raises(RuntimeError, match='원본을 보존'):
        store.restore_epoch(None, None, None)
    assert store.latest_checkpoint.read_bytes() == b'previous'
    model, _ = s.build_model(hm, 'M1')
    optimizer = torch.optim.Adam(model.parameters())
    monkeypatch.setattr(s.execution.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('Drive fail')))
    with pytest.raises(OSError, match='Drive fail'):
        store.save_epoch(model, optimizer, np.random.default_rng(51), epoch=1, history=[{'epoch': 1}])
    assert store.latest_checkpoint.read_bytes() == b'previous'


def test_epoch300_resume_evaluates_once_without_training(hm, monkeypatch):
    model, _ = s.build_model(hm, 'M1')
    store = s.ProgressStore(hm['run_dir']/'progress', s.identity(hm, 'M1'))
    store.save_epoch(model, torch.optim.Adam(model.parameters(), lr=hm['cfg'].lr),
        np.random.default_rng(51), epoch=300, history=[{'epoch': i} for i in range(1, 301)])
    monkeypatch.setattr(s, 'sample_uniform_negative_matrix', lambda *a, **kw: pytest.fail('retraining'))
    calls = []
    def evaluate(*args):
        calls.append(True)
        return dict.fromkeys(hm['references']['metrics'], 1.)
    monkeypatch.setattr(s.base.fixed, '_evaluate', evaluate)
    s.run_arm(hm, 'M1')
    s.run_arm(hm, 'M1')
    assert len(calls) == 1


def test_reference_failure_prevents_expensive_prepare(hm, monkeypatch):
    monkeypatch.setattr(s.base, '_prepare_hm_inputs', lambda *a: pytest.fail('loading before preflight'))
    with pytest.raises(RuntimeError, match='원본'):
        s.prepare(replace(hm['cfg'], m2_reference=str(hm['run_dir']/'absent')))


@pytest.mark.parametrize('state', ['absent', 'none', 'nonempty'])
def test_notebook_optional_weight_and_seed(state):
    notebook = json.loads(Path(__file__).with_name('clv_hm_final8_seed51_colab.ipynb').read_text())
    cell, = [''.join(c['source']) for c in notebook['cells'] if 'prepared = screen.prepare(cfg)' in ''.join(c['source'])]
    from types import SimpleNamespace
    data = {'splits': {'test': {}}}
    if state != 'absent': data['loss_w'] = None if state == 'none' else [1.]
    calls = []
    def prepare(cfg):
        calls.append(True)
        return {'data': data, 'run_dir': 'unused'}
    scope = {'screen': SimpleNamespace(prepare=prepare), 'cfg': s.configure()}
    if state == 'nonempty':
        with pytest.raises(AssertionError): exec(compile(cell, '<preflight>', 'exec'), scope)
    else:
        exec(compile(cell, '<preflight>', 'exec'), scope)
    assert calls == [True]


@pytest.mark.parametrize('mismatch', [None, 'input', 'features', 'weights', 'graph'])
def test_prepare_fingerprints_once_no_training(hm, monkeypatch, mismatch):
    p = hm
    p['data'].update(train=None, data_stats={'test': 'lastweek'}, splits={'test': {}})
    p['base_cfg'] = {}
    audit = p['protocol']['m4_weight_audit']
    p.update(observed_weight_audit=audit, protocol={'dataset_initialization': {'test': True}})
    p2 = {'input_hash': p['input_hash'], 'data_stats': p['data']['data_stats'],
          'feature_sha256': p['feature_sha256'], 'split': 'lastweek'}
    p3 = {'m4_c_weight_audit': audit, 'm3_audit': {'cv': .2}, 'fixed': {}}
    if mismatch == 'input': p2['input_hash'] = 'other'
    if mismatch == 'features': p2['feature_sha256'] = 'other'
    if mismatch == 'weights': p3['m4_c_weight_audit'] = {'sha256': 'other'}
    refs = p['references'] | {'design': {'m2': {'protocol': p2}, 'm4': {'protocol': p3}}}
    monkeypatch.setattr(s, 'load_references', lambda cfg: refs)
    calls = []
    def prepare_inputs(cfg):
        calls.append(True)
        return p
    monkeypatch.setattr(s.base, '_prepare_hm_inputs', prepare_inputs)
    monkeypatch.setattr(s.base.hm, 'validate_split', lambda data: None)
    monkeypatch.setattr(s.base.hm.hm_graph, '_purchase_keyed', lambda data: None)
    monkeypatch.setattr(s.base.fixed.graph, 'centered_edge_signals', lambda *a: {
        'edge_users': p['data']['pos_key']//5, 'edge_items': p['data']['pos_key']%5})
    monkeypatch.setattr(s.base.fixed, 'build_graph', lambda *a: {
        'audit': {'cv': .3 if mismatch == 'graph' else .2}, 'beta': .1,
        'weights': np.ones(len(p['data']['pos_key']))})
    monkeypatch.setattr(s.execution, 'train', lambda *a, **kw: pytest.fail('training in prepare'))
    if mismatch:
        with pytest.raises(RuntimeError): s.prepare(p['cfg'])
    else:
        result = s.prepare(p['cfg'])
        assert calls == [True] and result['base_cfg']['SEED_LIST'] == [51]
        assert result['protocol']['models'] == s.IDS
        assert len(s.report(result)['absolute']) == 0
