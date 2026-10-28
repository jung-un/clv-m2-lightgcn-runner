from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import clv_hm_ngcf_final8_seed49 as s
import clv_hm_graphsage_training_io as optimized_io
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as reference_fixture


def test_seed50_changes_only_seed_and_keeps_separate_identity(prepared, monkeypatch):
    old = prepared['cfg']
    cfg = s.configure(out_dir=old.out_dir, m2_reference=old.m2_reference,
                      m4_reference=old.m4_reference, seed=50)
    s.validate_config(cfg)
    assert replace(cfg, seeds=(49,)) == old
    prepared['cfg'] = cfg
    ident = s.identity(prepared, 'M1')
    assert ident.seed == 50 and 'seed50' in ident.stage
    assert s.arm_path(prepared, 'M1').name.endswith('_s50.json')
    seen = []
    def train(model, p, c, mid, seed, store, **kwargs):
        seen.append(seed)
        return [{'epoch': i} for i in range(1, 301)]
    monkeypatch.setattr(s.execution, 'train', train)
    monkeypatch.setattr(s.base.fixed, '_evaluate', lambda *a: dict.fromkeys(prepared['references']['metrics'], 1.))
    monkeypatch.setattr(s.execution, 'compact_completed_checkpoint', lambda *a: None)
    row = s.run_arm(prepared, 'M1')
    assert seen == [50] and row['seed'] == row['identity']['seed'] == 50
    assert s.completed(prepared, 'M1')['seed'] == 50
    assert set(s.report(prepared)['absolute']['seed']) == {50}
    prepared['cfg'] = old
    assert s.completed(prepared, 'M1') is None


def graphsage_train(*args, **kwargs):
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
    return p

@pytest.mark.parametrize('label', s.ORDER)
def test_eight_arms_routing_and_joint_gradients(prepared, label):
    s.base.v3.set_seed(49)
    model, weights = s.build_model(prepared, label)
    m2, m3, m4 = s.ARMS[label]
    expected = prepared['observed_adj'] if m3 else prepared['data']['adj']
    torch.testing.assert_close(model.adj.to_dense(), expected.to_dense())
    assert (weights is prepared['observed_weights']) if m4 else weights is None
    model.eval()
    users, items, _, _ = model.embeddings()
    assert users.shape == (3, 192) and items.shape == (5, 192)
    u, pos, neg = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([[4], [4], [0]])
    w = torch.tensor(weights[:3], dtype=torch.float32) if m4 else None
    loss, _, _ = s.base.fixed.recheck._batch_loss(model, u, pos, neg, w)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.E_u.weight.grad.norm() > 0
    assert all(layer.weight.grad.norm() > 0 for layer in list(model.sum_layers) + list(model.bi_layers))
    if m2:
        assert model.axis_logits.grad.norm() > 0 and model.rho_logit.grad.abs() > 0
        assert all(layer.weight.grad.norm() > 0 for layer in model.encoders.values())
    else:
        assert not hasattr(model, 'encoders') and not hasattr(model, 'axis_logits')


def test_resume_restores_optimizer_and_rng_exactly(prepared, tmp_path):
    cfg = replace(prepared['cfg'], epochs=2, batch_size=3)
    ident = s.identity(prepared, 'M5_D')
    def fresh():
        s.base.v3.set_seed(49)
        m, w = s.build_model(prepared, 'M5_D')
        s.base.v3.set_seed(49)
        return m, w
    full, weights = fresh()
    history = graphsage_train(full, prepared, cfg, ident.model_id, 49,
        s.SafeProgressStore(tmp_path / 'full', ident), row_weights=weights)
    part, _ = fresh()
    store = s.SafeProgressStore(tmp_path / 'resume', ident)
    graphsage_train(part, prepared, replace(cfg, epochs=1), ident.model_id, 49,
                        store, row_weights=weights)
    resumed, _ = fresh()
    actual = graphsage_train(resumed, prepared, cfg, ident.model_id, 49,
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
    for kwargs in ({'epochs': 100}, {'seeds': (51,)}, {'input_days': 704}, {'lr': .01}):
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
                                       ('optimized', graphsage_train, s.SafeProgressStore)]:
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
    graphsage_train(model, prepared, cfg, ident.model_id, 49, store, row_weights=weights)
    timing_path = store.root / 'timing' / f'{ident.model_id}_s49.csv'
    row = pd.read_csv(timing_path).iloc[0]
    assert row['profile_batches'] == 2 and row['epoch'] == 1
    assert row['checkpoint_drive_seconds'] > 0 and row['save_seconds'] > 0
    with ZipFile(s.export_zip(prepared, prepared['run_dir'] / 'timing.zip')) as archive:
        assert 'timing/' + timing_path.name in archive.namelist()


@pytest.mark.parametrize('out', ['/tmp/graphsage_refs', '/tmp/graphsage_refs/child', '/tmp'])
def test_reject_output_reference_overlap(out):
    with pytest.raises(ValueError, match='분리'):
        s.validate_config(s.configure(out_dir=out, m2_reference='/tmp/graphsage_refs',
                                      m4_reference='/tmp/graphsage_other'))


def test_final_checkpoint_restoration_does_not_repeat_training(prepared, monkeypatch):
    model, weights = s.build_model(prepared, 'M5_D')
    ident = s.identity(prepared, 'M5_D')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', ident)
    history = [{'epoch': e, 'loss': 1., 'p_correct': .5} for e in range(1, 301)]
    store.save_epoch(model, torch.optim.Adam(model.parameters(), lr=prepared['cfg'].lr),
                     np.random.default_rng(49), epoch=300, history=history)
    monkeypatch.setattr(model, '_pair_scores', lambda *a: pytest.fail('trained past epoch300'))
    assert graphsage_train(model, prepared, prepared['cfg'], ident.model_id, 49, store,
                     row_weights=weights) == history


def test_hm_configuration_and_real_reference_files():
    cfg = s.configure()
    assert cfg.dataset == 'hm' and cfg.seeds == (49,) and cfg.batch_size == 131072
    assert cfg.epochs == 300 and cfg.negative_count == 1
    with pytest.raises(ValueError):
        s.validate_config(replace(cfg, batch_size=8192))


def test_hm_prepare_routes_inputs_and_graph_without_training(prepared, monkeypatch):
    p = prepared
    p.update(q_n=np.full(3,.5,np.float32), q_v=np.full(3,.5,np.float32),
             item_amount_percentile=np.full(5,.5,np.float32), observed_weight_audit={'sha256':'weight'})
    p['data'].update(splits={'test':{}}, data_stats={'same':True}, train={})
    p['base_cfg'] = {}
    gc = s.base.hm.configure()
    audit = dict(coefficient_of_variation=.2, customer_weight_vs_degree=0.,
                 item_weight_vs_price=0., item_weight_vs_popularity=0.,
                 max_customer_mass_error=0., weight_min=.8, weight_max=1.2)
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
    torch.testing.assert_close(actual['observed_adj'].to_dense(),p['observed_adj'].to_dense())
    assert s.report(actual)['absolute'].empty


def test_input_bytes_checked_before_expensive_preparation(tmp_path, monkeypatch):
    tx, meta = tmp_path/'tx.parquet', tmp_path/'articles.csv'
    tx.write_bytes(b'original parquet bytes')
    meta.write_bytes(b'article_id,category\n0123,A\n')
    monkeypatch.setitem(s.base.v3.SCHEMA, 'hm', {'tx_path':str(tx),'item_meta_path':str(meta)})
    expected = s.base.fixed.moe.manifest_hash(s.base.fixed.moe.build_input_manifest(s.base.v3.SCHEMA['hm']))
    assert s.verify_input_files(expected)['transactions']['bytes'] == tx.stat().st_size
    tx.write_bytes(b'reencoded but different parquet')
    with pytest.raises(RuntimeError, match='원본 파일 불일치'):
        s.verify_input_files(expected)
    refs={'sources':{'m2':{'protocol':{'input_hash':expected}},'m4':{'protocol':{'config':{k:getattr(s.base.hm.configure(),k) for k in s.hm_schema.GRAPH_KEYS}}}}}
    monkeypatch.setattr(s.base, 'load_references', lambda cfg:refs)
    monkeypatch.setattr(s.base, '_prepare_hm_inputs', lambda cfg:pytest.fail('expensive preparation started'))
    with pytest.raises(RuntimeError, match='원본 파일 불일치'):
        s.prepare(s.configure())


def test_completed_checkpoint_compaction_preserves_weights_and_failure_safety(prepared, monkeypatch):
    import os
    from clv_hm_graphsage_training_io import compact_completed_checkpoint
    model, _ = s.build_model(prepared, 'M1')
    store = s.SafeProgressStore(prepared['run_dir']/'compact',s.identity(prepared,'M1'))
    optimizer, rng = torch.optim.Adam(model.parameters()), np.random.default_rng(49)
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    history=[{'epoch': i} for i in range(1,301)]
    store.save_epoch(model,optimizer,rng,epoch=300,history=history)
    original=store.latest_checkpoint.read_bytes()
    compact_completed_checkpoint(store)
    assert store.latest_checkpoint.read_bytes()==original  # not completed yet
    result=store.root/'result.json';result.write_text('{}')
    store.mark_complete(epoch=300,result_path=str(result))
    real_replace=os.replace
    monkeypatch.setattr(os,'replace',lambda *a: (_ for _ in ()).throw(OSError('publish failure')))
    with pytest.raises(OSError,match='publish failure'):
        compact_completed_checkpoint(store)
    assert store.latest_checkpoint.read_bytes()==original
    monkeypatch.setattr(os,'replace',real_replace)
    compact_completed_checkpoint(store)
    payload=torch.load(store.latest_checkpoint,weights_only=False)
    assert payload['completed_weights_only'] and 'optimizer_state' not in payload
    assert payload['history']==history
    assert payload['numpy_rng_state']==rng.bit_generator.state
    for key,value in model.state_dict().items():
        torch.testing.assert_close(payload['model_state'][key],value,rtol=0,atol=0)
    assert store.latest_checkpoint.stat().st_size < len(original)
    compact_completed_checkpoint(store)  # idempotent
    with pytest.raises(RuntimeError,match='최종 가중치'):
        store.restore_epoch(model,optimizer,rng)



def test_storage_guard_blocks_publish_before_touching_existing_checkpoint(prepared, monkeypatch):
    model, _ = s.build_model(prepared, 'M1')
    store = s.SafeProgressStore(prepared['run_dir'] / 'progress', s.identity(prepared, 'M1'))
    store.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    store.latest_checkpoint.write_bytes(b'previous checkpoint')
    required = []
    def full_disk(size):
        required.append(size)
        raise RuntimeError('quota')
    monkeypatch.setattr(s, 'STORAGE_GUARD', full_disk)
    with pytest.raises(RuntimeError, match='quota'):
        store.save_epoch(model, None, None, epoch=1)
    assert store.latest_checkpoint.read_bytes() == b'previous checkpoint'
    assert required == [sum(p.numel()*p.element_size() for p in model.parameters())*3 + 512*1024**2]
