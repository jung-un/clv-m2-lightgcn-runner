from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import clv_dh_gat_final8_seed49 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as reference_fixture


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
    p.update(s.prepare_attention_priors(p))
    return p


@pytest.mark.parametrize('label', s.ORDER)
def test_eight_arms_routing_and_joint_gradients(prepared, label):
    s.base.v3.set_seed(49)
    model, weights = s.build_model(prepared, label)
    m2, m3, m4 = s.ARMS[label]
    expected = prepared['gat_m3_prior'] if m3 else prepared['gat_binary_prior']
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
    assert all(layer.projection.weight.grad.norm() > 0 for layer in model.gat_layers)
    if m2:
        assert model.axis_logits.grad.norm() > 0 and model.rho_logit.grad.abs() > 0
        assert all(layer.weight.grad.norm() > 0 for layer in model.encoders.values())
    else:
        assert not hasattr(model, 'encoders') and not hasattr(model, 'axis_logits')


def test_common_initialization_and_dense_gat_formula(prepared):
    initial = None
    for label in s.ORDER:
        s.base.v3.set_seed(49)
        model, _ = s.build_model(prepared, label)
        state = {k: v for k, v in model.state_dict().items() if k.startswith(('E_', 'gat_layers'))}
        if initial is None:
            initial = state
        for key in initial:
            torch.testing.assert_close(state[key], initial[key], rtol=0, atol=0)
    model.eval()
    x = torch.cat((model.E_u.weight, model.E_i.weight))
    layers = [x]
    for layer in model.gat_layers:
        h = layer.projection(x)
        logits = F.leaky_relu((h @ layer.receiver_attention)[:, None]
                             + (h @ layer.source_attention)[None, :], .2)
        prior = model.adj.to_dense()
        active = prior.sum(1) > 0
        alpha = torch.zeros_like(prior)
        alpha[active] = torch.softmax((logits + prior.clamp_min(1e-30).log())
                                      .masked_fill(prior == 0, -torch.inf)[active], dim=1)
        x = F.elu(alpha @ h)
        layers.append(x)
    users, items = model._id_embeddings()
    torch.testing.assert_close(torch.cat((users, items)), torch.stack(layers).mean(0))


def test_m3_is_raw_prior_times_attention_and_row_normalized(prepared):
    s.base.v3.set_seed(49)
    binary, _ = s.build_model(prepared, 'M1')
    s.base.v3.set_seed(49)
    weighted, _ = s.build_model(prepared, 'M3')
    x = torch.cat((binary.E_u.weight, binary.E_i.weight))
    layer = binary.gat_layers[0]
    receiver, source = binary.edge_receiver, binary.edge_source
    alpha, _ = layer.attention_weights(x, receiver, source, len(x), binary.edge_log_prior)
    actual, _ = weighted.gat_layers[0].attention_weights(
        x, receiver, source, len(x), weighted.edge_log_prior)
    product = alpha * weighted.edge_log_prior.exp()
    row_total = product.new_zeros(len(x)).scatter_add_(0, receiver, product)
    torch.testing.assert_close(actual, product / row_total[receiver])
    row_sum = actual.new_zeros(len(x)).scatter_add_(0, receiver, actual)
    torch.testing.assert_close(row_sum[row_total > 0], torch.ones_like(row_sum[row_total > 0]))
    assert not torch.allclose(alpha, actual)
    # A common scale per receiver cancels: raw LightGCN row mass is not retained.
    scaled, _ = layer.attention_weights(x, receiver, source, len(x),
        weighted.edge_log_prior + torch.arange(len(x), dtype=x.dtype)[receiver])
    torch.testing.assert_close(actual, scaled)


def test_recommendation_gradient_reaches_m2_and_gat_without_l2(prepared):
    s.base.v3.set_seed(49)
    model, _ = s.build_model(prepared, 'M5_D')
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([4, 4, 0])
    positive, negative = model._pair_scores(u, pos, neg)
    F.softplus(negative - positive).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_resume_restores_optimizer_and_dropout_rng_exactly(prepared, tmp_path):
    cfg = replace(prepared['cfg'], epochs=2, batch_size=3)
    ident = s.identity(prepared, 'M5_D')
    def fresh():
        s.base.v3.set_seed(49)
        m, w = s.build_model(prepared, 'M5_D')
        s.base.v3.set_seed(49)
        return m, w
    full, weights = fresh()
    history = s.execution.train(full, prepared, cfg, ident.model_id, 49,
        s.SafeProgressStore(tmp_path / 'full', ident), row_weights=weights)
    part, _ = fresh()
    store = s.SafeProgressStore(tmp_path / 'resume', ident)
    s.execution.train(part, prepared, replace(cfg, epochs=1), ident.model_id, 49,
                        store, row_weights=weights)
    resumed, _ = fresh()
    actual = s.execution.train(resumed, prepared, cfg, ident.model_id, 49,
                                store, row_weights=weights)
    assert actual == history
    for key, value in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    assert s.SafeProgressStore(tmp_path / 'resume', ident).latest_checkpoint.is_file()


def test_completed_arms_skipped_and_partial_reports(prepared, monkeypatch):
    fits, evals = [], []
    def train(model, p, cfg, model_id, seed, store, row_weights=None):
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
                                       ('optimized', s.execution.train, s.SafeProgressStore)]:
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
    s.execution.train(model, prepared, cfg, ident.model_id, 49, store, row_weights=weights)
    timing_path = store.root / 'timing' / f'{ident.model_id}_s49.csv'
    row = pd.read_csv(timing_path).iloc[0]
    assert row['profile_batches'] == 2 and row['epoch'] == 1
    assert row['checkpoint_drive_seconds'] > 0 and row['save_seconds'] > 0
    with ZipFile(s.export_zip(prepared, prepared['run_dir'] / 'timing.zip')) as archive:
        assert 'timing/' + timing_path.name in archive.namelist()
