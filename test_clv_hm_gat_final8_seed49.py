from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import clv_hm_gat_final8_seed49 as s
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as reference_fixture


def gat_train(*args, **kwargs):
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
    history = gat_train(full, prepared, cfg, ident.model_id, 49,
        s.SafeProgressStore(tmp_path / 'full', ident), row_weights=weights)
    part, _ = fresh()
    store = s.SafeProgressStore(tmp_path / 'resume', ident)
    gat_train(part, prepared, replace(cfg, epochs=1), ident.model_id, 49,
                        store, row_weights=weights)
    resumed, _ = fresh()
    actual = gat_train(resumed, prepared, cfg, ident.model_id, 49,
                                store, row_weights=weights)
    assert actual == history
    for key, value in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    assert s.SafeProgressStore(tmp_path / 'resume', ident).latest_checkpoint.is_file()


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
    result = s.run(prepared, models=s.ORDER)
    assert result['complete'] and list(result['absolute'].model) == list(s.ORDER)
    assert len(result['comparison']) == 8 * 7 * 142
    s.run(prepared, models=s.ORDER)
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




@pytest.mark.parametrize('label', s.ORDER)
def test_optimized_training_exactly_matches_original(prepared, tmp_path, label):
    cfg = replace(prepared['cfg'], epochs=2, batch_size=4)
    ident = s.identity(prepared, label)
    outcomes = []
    for name, trainer, store_type in [('original', s.base.fixed._train, s.base.ProgressStore),
                                       ('optimized', gat_train, s.SafeProgressStore)]:
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
    import clv_ngcf_training_io as shared_io
    monkeypatch.setattr(shared_io.os, 'replace', fail)
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
    gat_train(model, prepared, cfg, ident.model_id, 49, store, row_weights=weights)
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
        gat_train(model, prepared, cfg, ident.model_id, 49, store)
    assert not store.latest_checkpoint.exists()


def test_default_scope_is_only_m1_to_m4(prepared, monkeypatch):
    calls=[]
    monkeypatch.setattr(s, 'run_arm', lambda p,m: calls.append(m))
    s.run(prepared)
    assert calls == ['M1','M2','M3','M4']
    assert prepared['cfg'].dataset=='hm' and prepared['cfg'].batch_size==131072
    assert prepared['cfg'].feature_dropout==0


def test_same_gat_model_and_sampler():
    assert s.build_model is s.gat_core.build_model
    assert s.sample_uniform_negative_matrix is s.gat_core.sample_uniform_negative_matrix


def test_hm_prepare_routes_inputs_and_graph_without_training(prepared, monkeypatch):
    p = prepared
    p.update(q_n=np.full(3,.5,np.float32), q_v=np.full(3,.5,np.float32),
             item_amount_percentile=np.full(5,.5,np.float32), observed_weight_audit={'sha256':'weight'})
    p['data'].update(splits={'test':{}}, data_stats={'same':True}, train={})
    p['base_cfg'] = {}
    audit = dict(coefficient_of_variation=.2, customer_weight_vs_degree=.01, item_weight_vs_price=.02, item_weight_vs_popularity=.03, max_customer_mass_error=1e-12, weight_min=.7, weight_max=1.3)
    gc = s.base.hm.configure()
    common = {'input_hash':p['input_hash'], 'data_stats':p['data']['data_stats'],
              'feature_sha256':p['feature_sha256'], 'split':'hm-lastweek',
              'm4_c_weight_audit':{'sha256':'weight'}, 'm3_audit':audit, 'fixed':{},
              'config':{k:getattr(gc,k) for k in s.hm_schema.GRAPH_KEYS}}
    refs = {**p['references'], 'sources':{k:{'protocol':common} for k in ('m2','m4')},'audit':{}}
    monkeypatch.setattr(s.base,'load_references',lambda cfg:refs)
    monkeypatch.setattr(s,'verify_input_files',lambda expected: {})
    monkeypatch.setattr(s.base,'_prepare_hm_inputs',lambda cfg:dict(p))
    monkeypatch.setattr(s.base.hm,'validate_split',lambda data:None)
    monkeypatch.setattr(s.base.hm.hm_graph,'_purchase_keyed',lambda data:None)
    monkeypatch.setattr(s.base.fixed.graph,'centered_edge_signals',lambda *a:p['signals'])
    monkeypatch.setattr(s.base.fixed,'build_graph',lambda *a:{**p['observed_graph'],'audit':audit})
    monkeypatch.setattr(s.execution,'train',lambda *a,**kw:pytest.fail('prepare trained'))
    monkeypatch.setattr(s.base.fixed,'_evaluate',lambda *a:pytest.fail('prepare evaluated'))
    actual=s.prepare(p['cfg'])
    assert actual['cfg'].dataset=='hm' and actual['base_cfg']['SEED_LIST']==[49]
    assert actual['protocol']['new_fit_count']==8
    assert actual['observed_weights'] is p['observed_weights']
    assert actual['features'] is p['features']
    torch.testing.assert_close(actual['gat_m3_prior'].to_dense(),p['gat_m3_prior'].to_dense())
    assert actual['protocol']['default_execution_models']==['M1','M2','M3','M4']
    assert actual['protocol']['backbone'].startswith('GAT:')
    assert s.report(actual)['absolute'].empty

