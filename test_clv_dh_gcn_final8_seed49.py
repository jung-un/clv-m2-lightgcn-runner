from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import clv_dh_gcn_final8_seed49 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as reference_fixture


def gcn_train(*args, **kwargs):
    return s.execution.train(*args, negative_sampler=s.sample_uniform_negative_matrix, **kwargs)


@pytest.fixture
def prepared(reference_fixture):
    p = reference_fixture
    p['cfg'] = s.configure(out_dir=str(p['run_dir']))
    p['data'].update(tr_u=np.array([0, 0, 1, 1, 2, 2]),
                     tr_i=np.array([0, 1, 1, 2, 2, 3]),
                     pos_key=np.array([0, 1, 6, 7, 12, 13]))
    for name in ('protocol.json', 'reference_audit.json'):
        (p['run_dir'] / name).write_text('{}')
    p['signals'] = {'edge_users': np.array([0, 0, 1, 1, 2, 2]),
                    'edge_items': np.array([0, 1, 1, 2, 2, 3])}
    p['observed_graph'] = {'weights': np.array([.8, 1.2, 1.1, .9, .7, 1.3])}
    p.update(s.prepare_gcn_graphs(p))
    return p


@pytest.mark.parametrize('label', s.ORDER)
def test_eight_arms_routing_and_joint_gradients(prepared, label):
    s.base.v3.set_seed(49)
    model, weights = s.build_model(prepared, label)
    m2, m3, m4 = s.ARMS[label]
    expected = prepared['gcn_m3_adj'] if m3 else prepared['gcn_binary_adj']
    torch.testing.assert_close(model.adj.to_dense(), expected.to_dense())
    assert (weights is prepared['observed_weights']) if m4 else weights is None
    model.eval()
    users, items, _, _ = model.embeddings()
    assert users.shape == (3, 64) and items.shape == (5, 64)
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([[4], [4], [0]])
    w = torch.tensor(weights[:3], dtype=torch.float32) if m4 else None
    loss, _, _ = s.base.fixed.recheck._batch_loss(model, u, pos, neg, w)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.E_u.weight.grad.norm() > 0
    assert all(layer.weight.grad.norm() > 0 for layer in model.gcn_layers)
    if m2:
        assert model.axis_logits.grad.norm() > 0 and model.rho_logit.grad.abs() > 0
        assert all(layer.weight.grad.norm() > 0 for layer in model.encoders.values())
    else:
        assert not hasattr(model, 'encoders') and not hasattr(model, 'axis_logits')


def test_common_initialization_and_dense_gcn_formula(prepared):
    initial = None
    for label in s.ORDER:
        s.base.v3.set_seed(49)
        model, _ = s.build_model(prepared, label)
        state = {k: v for k, v in model.state_dict().items() if k.startswith(('E_', 'gcn_layers'))}
        if initial is None:
            initial = state
        for key in initial:
            torch.testing.assert_close(state[key], initial[key], rtol=0, atol=0)
    x = torch.cat((model.E_u.weight, model.E_i.weight))
    dense = model.adj.to_dense()
    # Non-zero biases detect the incorrect S(HW+b) ordering.
    with torch.no_grad():
        model.gcn_layers[0].bias.fill_(.01)
        model.gcn_layers[1].bias.fill_(.02)
    x = F.relu(F.linear(dense @ x, model.gcn_layers[0].weight, model.gcn_layers[0].bias))
    expected = F.linear(dense @ x, model.gcn_layers[1].weight, model.gcn_layers[1].bias)
    users, items = model._id_embeddings()
    torch.testing.assert_close(torch.cat((users, items)), expected)


def test_m3_raw_weights_self_loops_then_symmetric_normalization(prepared):
    graphs = s.prepare_gcn_graphs(prepared)
    for name in ('gcn_binary_adj', 'gcn_m3_adj'):
        raw = torch.eye(8)
        weights = prepared['observed_graph']['weights'] if name == 'gcn_m3_adj' else np.ones(6)
        for u, i, w in zip(prepared['signals']['edge_users'], prepared['signals']['edge_items'], weights):
            raw[u, i + 3] = raw[i + 3, u] = float(w)
        inv = raw.sum(1).rsqrt()
        expected = inv[:, None] * raw * inv[None, :]
        torch.testing.assert_close(graphs[name].to_dense(), expected)
        torch.testing.assert_close(graphs[name].to_dense(), graphs[name].to_dense().T)
        assert (graphs[name].to_dense().diag() > 0).all()
        assert graphs[name].to_dense()[7, 7] == 1  # isolated catalog item retains self
    assert not torch.allclose(graphs['gcn_binary_adj'].to_dense(), graphs['gcn_m3_adj'].to_dense())
    prepared['observed_graph']['weights'] = np.ones(6)
    equal = s.prepare_gcn_graphs(prepared)
    torch.testing.assert_close(equal['gcn_binary_adj'].to_dense(), equal['gcn_m3_adj'].to_dense())


@pytest.mark.parametrize('bad', [np.zeros(6), np.full(6, np.nan), np.ones(5), -np.ones(6)])
def test_invalid_m3_weight_fails_before_training(prepared, bad):
    prepared['observed_graph']['weights'] = bad
    with pytest.raises(RuntimeError):
        s.prepare_gcn_graphs(prepared)


def test_recommendation_gradient_reaches_m2_and_gcn_without_l2(prepared):
    s.base.v3.set_seed(49)
    model, _ = s.build_model(prepared, 'M5_D')
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([4, 4, 0])
    positive, negative = model._pair_scores(u, pos, neg)
    F.softplus(negative - positive).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_resume_restores_optimizer_and_rng_exactly(prepared, tmp_path):
    cfg = replace(prepared['cfg'], epochs=2, batch_size=3)
    ident = s.identity(prepared, 'M5_D')
    def fresh():
        s.base.v3.set_seed(49)
        m, w = s.build_model(prepared, 'M5_D')
        s.base.v3.set_seed(49)
        return m, w
    full, weights = fresh()
    history = gcn_train(full, prepared, cfg, ident.model_id, 49,
        s.SafeProgressStore(tmp_path / 'full', ident), row_weights=weights)
    part, _ = fresh()
    store = s.SafeProgressStore(tmp_path / 'resume', ident)
    gcn_train(part, prepared, replace(cfg, epochs=1), ident.model_id, 49,
                        store, row_weights=weights)
    resumed, _ = fresh()
    actual = gcn_train(resumed, prepared, cfg, ident.model_id, 49,
                                store, row_weights=weights)
    assert actual == history
    for key, value in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    saved_full = torch.load(s.SafeProgressStore(tmp_path / 'full', ident).latest_checkpoint,
                            weights_only=False)
    saved_resume = torch.load(store.latest_checkpoint, weights_only=False)
    assert saved_full['numpy_rng_state'] == saved_resume['numpy_rng_state']
    torch.testing.assert_close(saved_full['torch_rng_state'], saved_resume['torch_rng_state'],
                               rtol=0, atol=0)
    for key, values in saved_full['optimizer_state']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(saved_resume['optimizer_state']['state'][key][name],
                                       value, rtol=0, atol=0)


def test_completed_arms_skipped_and_partial_reports(prepared, monkeypatch):
    fits, evals = [], []
    def train(model, p, cfg, model_id, seed, store, row_weights=None, *, negative_sampler=None):
        assert negative_sampler is s.sample_uniform_negative_matrix
        assert cfg.epochs == 300 and seed == 49
        fits.append(model_id)
        return [{'epoch': i} for i in range(1, 301)]
    def evaluate(model, p):
        evals.append(True)
        return dict.fromkeys(p['references']['metrics'], 1.01)
    monkeypatch.setattr(s.execution, 'train', train)
    monkeypatch.setattr(s.base.fixed, '_evaluate', evaluate)
    assert s.report(prepared)['absolute'].empty  # no LightGCN metrics mixed in
    s.run_arm(prepared, 'M1')
    path = s.export_zip(prepared, prepared['run_dir'] / 'result.zip')
    assert path.endswith('_partial.zip')
    with ZipFile(path) as archive:
        assert len([n for n in archive.namelist() if n.startswith('arms/')]) == 1
    result = s.run(prepared)
    assert result['complete'] and list(result['absolute'].model) == list(s.ORDER)
    assert len(result['comparison']) == 8 * 7 * 142
    s.run(prepared)
    assert len(fits) == len(evals) == 8
    row = json.loads(s.arm_path(prepared, 'M1').read_text())
    row['identity']['seed'] = 50
    s.arm_path(prepared, 'M1').write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match='신원'):
        s.run_arm(prepared, 'M1')


def test_incomplete_epoch_and_changed_config_never_evaluate(prepared, monkeypatch):
    monkeypatch.setattr(s.execution, 'train', lambda *a, **kw: [{'epoch': 299}])
    monkeypatch.setattr(s.base.fixed, '_evaluate', lambda *a: pytest.fail('early test'))
    with pytest.raises(RuntimeError, match='300epoch'):
        s.run_arm(prepared, 'M1')
    for kwargs in ({'epochs': 100}, {'seeds': (50,)}, {'input_days': 704}, {'lr': .01}):
        with pytest.raises(ValueError):
            s.validate_config(replace(prepared['cfg'], **kwargs))


def test_corrupt_checkpoint_is_kept_and_not_restarted(prepared):
    model, _ = s.build_model(prepared, 'M1')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', s.identity(prepared, 'M1'))
    store.latest_checkpoint.parent.mkdir(parents=True)
    store.latest_checkpoint.write_bytes(b'broken')
    with pytest.raises(RuntimeError, match='파일은 보존'):
        store.restore_epoch(model, torch.optim.Adam(model.parameters()), np.random.default_rng(49))
    assert store.latest_checkpoint.read_bytes() == b'broken'


def test_prepare_keeps_clv_inputs_and_does_not_fit(prepared, monkeypatch):
    original = dict(prepared)
    original['protocol'] = {'split': 'train DAY 1-704; test DAY 705-711',
                            'm3_audit': {'cv': .2}, 'm4_weight_audit': {'mean': 1},
                            'fixed': {'clv_days': 365}}
    original['data'].update(train={'b_raw': [1]}, splits={'test': {}},
                            loss_w=None, data_stats={'marker': 'same'})
    monkeypatch.setattr(s.base, 'prepare', lambda cfg: dict(original))
    monkeypatch.setattr(s.base.prior.lastweek, 'validate_split', lambda *a: None)
    monkeypatch.setattr(s.execution, 'train', lambda *a: pytest.fail('prepare trained'))
    monkeypatch.setattr(s.base.fixed, '_evaluate', lambda *a: pytest.fail('prepare evaluated'))
    actual = s.prepare(prepared['cfg'])
    assert actual['features'] is original['features']
    assert actual['observed_adj'] is original['observed_adj']
    assert actual['observed_weights'] is original['observed_weights']
    assert actual['protocol']['new_fit_count'] == 8
    assert actual['protocol']['m3_audit'] == original['protocol']['m3_audit']
    assert s.report(actual)['absolute'].empty


@pytest.mark.parametrize('label', s.ORDER)
def test_optimized_training_exactly_matches_original(prepared, tmp_path, label):
    cfg = replace(prepared['cfg'], epochs=2, batch_size=4)
    ident = s.identity(prepared, label)
    outcomes = []
    for name, trainer, store_type in [('original', s.base.fixed._train, s.base.ProgressStore),
                                       ('optimized', gcn_train, s.SafeProgressStore)]:
        s.base.v3.set_seed(49)
        model, weights = s.build_model(prepared, label)
        s.base.v3.set_seed(49)
        store = store_type(tmp_path / name, ident)
        hist = trainer(model, prepared, cfg, ident.model_id, 49, store, row_weights=weights)
        payload = torch.load(store.latest_checkpoint, weights_only=False)
        outcomes.append((hist, model.state_dict(), payload))
    assert outcomes[0][0] == outcomes[1][0]
    for key, value in outcomes[0][1].items():
        torch.testing.assert_close(outcomes[1][1][key], value, rtol=0, atol=0)
    assert outcomes[0][2]['numpy_rng_state'] == outcomes[1][2]['numpy_rng_state']
    torch.testing.assert_close(outcomes[0][2]['torch_rng_state'], outcomes[1][2]['torch_rng_state'], rtol=0, atol=0)
    for key, value in outcomes[0][2]['optimizer_state']['state'].items():
        for name, tensor in value.items():
            torch.testing.assert_close(outcomes[1][2]['optimizer_state']['state'][key][name], tensor, rtol=0, atol=0)


def test_drive_publish_failure_preserves_previous_checkpoint(prepared, monkeypatch):
    import hashlib
    model, _ = s.build_model(prepared, 'M1')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', s.identity(prepared, 'M1'))
    optimizer, rng = torch.optim.Adam(model.parameters()), np.random.default_rng(49)
    store.save_epoch(model, optimizer, rng, epoch=1, history=[{'epoch': 1}])
    original = store.latest_checkpoint.read_bytes()
    assert store.read_progress()['checkpoint_sha256'] == hashlib.sha256(original).hexdigest()
    def fail(*args):
        raise OSError('simulated Drive disconnected')
    monkeypatch.setattr(s.execution.os, 'replace', fail)
    with pytest.raises(OSError, match='Drive disconnected'):
        store.save_epoch(model, optimizer, rng, epoch=2, history=[{'epoch': 1}, {'epoch': 2}])
    assert store.latest_checkpoint.read_bytes() == original
    assert store.read_progress()['epoch'] == 1


def test_timing_csv_and_zip_without_extra_evaluation(prepared):
    import pandas as pd
    cfg = replace(prepared['cfg'], epochs=1, batch_size=4)
    model, weights = s.build_model(prepared, 'M5_D')
    ident = s.identity(prepared, 'M5_D')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', ident)
    gcn_train(model, prepared, cfg, ident.model_id, 49, store, row_weights=weights)
    timing_path = store.root / 'timing' / f'{ident.model_id}_s49.csv'
    row = pd.read_csv(timing_path).iloc[0]
    assert row['profile_batches'] == 2 and row['epoch'] == 1
    assert row['checkpoint_drive_seconds'] > 0 and row['save_seconds'] > 0
    with ZipFile(s.export_zip(prepared, prepared['run_dir'] / 'timing.zip')) as archive:
        assert 'timing/' + timing_path.name in archive.namelist()


def test_fast_negative_validation_preserves_draws_dtype_and_rng():
    rng = np.random.default_rng(31)
    n_users, n_items = 32, 300
    keys = np.sort(rng.choice(n_users * n_items, 2100, replace=False))
    pairs = keys[rng.integers(0, len(keys), 128)]
    users, positives = pairs // n_items, pairs % n_items
    old_rng, new_rng = np.random.default_rng(49), np.random.default_rng(49)
    for _ in range(30):
        expected = s.execution.fixed.m4_helpers.sample_uniform_negative_matrix(
            users, positives, n_items, keys, old_rng, k=1)
        actual = s.sample_uniform_negative_matrix(users, positives, n_items, keys, new_rng, k=1)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (128, 1) and actual.dtype == expected.dtype
        assert new_rng.bit_generator.state == old_rng.bit_generator.state
        assert not np.isin(users[:, None] * n_items + actual, keys).any()


def test_fast_negative_validation_boundaries_and_observed_pairs(monkeypatch):
    users, positives = np.array([0, 2]), np.array([1, 3])
    keys = np.array([1, 13])
    monkeypatch.setattr(s.base.v3, 'sample_negatives', lambda *a: np.array([0, 4]))
    actual = s.sample_uniform_negative_matrix(users, positives, 5, keys, np.random.default_rng(49), k=1)
    np.testing.assert_array_equal(actual[:, 0], [0, 4])  # below first / above last key
    monkeypatch.setattr(s.base.v3, 'sample_negatives', lambda *a: np.array([1, 3]))
    with pytest.raises(RuntimeError, match='학습 관측상품'):
        s.sample_uniform_negative_matrix(users, positives, 5, keys, np.random.default_rng(49), k=1)
    with pytest.raises(ValueError, match='K1'):
        s.sample_uniform_negative_matrix(users, positives, 5, keys, np.random.default_rng(49), k=2)


@pytest.mark.parametrize('keys', [np.array([], dtype=np.int64), np.array([1, 1]), np.array([2, 1])])
def test_fast_training_rejects_invalid_key_order_before_fitting(prepared, keys):
    model, _ = s.build_model(prepared, 'M1')
    prepared['data']['pos_key'] = keys
    cfg = replace(prepared['cfg'], epochs=1, batch_size=4)
    ident = s.identity(prepared, 'M1')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', ident)
    with pytest.raises(ValueError, match='정렬된 고유'):
        gcn_train(model, prepared, cfg, ident.model_id, 49, store)
    assert not store.latest_checkpoint.exists()


@pytest.mark.parametrize('out', ['/tmp/gcn_refs', '/tmp/gcn_refs/child', '/tmp'])
def test_reject_output_reference_overlap(out):
    with pytest.raises(ValueError, match='분리'):
        s.validate_config(s.configure(out_dir=out, m2_reference='/tmp/gcn_refs',
                                      m4_reference='/tmp/gcn_other'))


def test_final_checkpoint_restoration_does_not_repeat_training(prepared, monkeypatch):
    model, weights = s.build_model(prepared, 'M5_D')
    ident = s.identity(prepared, 'M5_D')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', ident)
    history = [{'epoch': e, 'loss': 1., 'p_correct': .5} for e in range(1, 301)]
    store.save_epoch(model, torch.optim.Adam(model.parameters(), lr=prepared['cfg'].lr),
                     np.random.default_rng(49), epoch=300, history=history)
    monkeypatch.setattr(model, '_pair_scores', lambda *a: pytest.fail('trained past epoch300'))
    assert gcn_train(model, prepared, prepared['cfg'], ident.model_id, 49, store,
                     row_weights=weights) == history
