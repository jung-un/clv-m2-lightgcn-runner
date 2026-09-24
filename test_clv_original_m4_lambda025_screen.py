from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

import clv_original_m4_lambda025_screen as screen
from clv_run_state import ProgressStore, RunIdentity
from test_linear_nv_original_m4_screen import inputs


def test_only_m4_lambda_changes_and_only_one_new_arm():
    cfg = screen.configure('unused')
    old = screen.previous.configure('unused')
    changed = {key for key in asdict(cfg) if getattr(cfg, key) != getattr(old, key)}
    assert changed == {'positive_weight_lambda'}
    assert cfg.positive_weight_lambda == .25
    assert screen.spec()['role'] == 'M4'
    assert screen.spec()['rho'] == 0
    assert screen.spec()['graph'] == 'binary'


def test_lambda025_keeps_original_valid_formula_and_masks_invalid_rows():
    prep = inputs()
    cfg = screen.configure('unused')
    users = prep['data']['tr_u']
    items = prep['data']['tr_i']
    raw = 1 + .25*prep['q_c'][users]*prep['item_amount_percentile'][items]
    _, weights, meta = screen.previous.audit_weights(prep, cfg)
    np.testing.assert_allclose(weights, raw/raw.mean())
    prep['item_economic_valid'][0] = False
    prep['item_bin'][0] = -1
    _, weights, meta = screen.previous.audit_weights(prep, cfg)
    invalid = ~prep['item_economic_valid'][items]
    corrected = weights*meta['train_mean_raw_weight']
    np.testing.assert_array_equal(corrected[invalid], np.ones(invalid.sum()))
    np.testing.assert_allclose(corrected[~invalid], raw[~invalid])
    assert meta['original_invalid_extra_absent']


def test_missing_exact_report_stops_before_data_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(screen.base, '_prepare', lambda *_: pytest.fail('must not load data'))
    with pytest.raises(ValueError, match='Exact completed lambda-0.5 report'):
        screen.prepare(tmp_path/'missing.json', tmp_path/'output')


def test_new_m4_uses_existing_joint_bpr_training_path(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(screen.base.v3, 'DEVICE', 'cpu')
    prep = inputs()
    prep['item_economic_valid'][0] = False
    prep['item_bin'][0] = -1
    cfg = replace(screen.configure(str(tmp_path)), epochs=2, eval_every=1,
                  patience_start=1, patience=1, batch_size=2)
    _, prep['m4_weights'], _ = screen.previous.audit_weights(prep, cfg)
    metrics = {key: 1. for key in (*screen.base.ACCURACY, *screen.es.fixed.PRIMARY)}
    monkeypatch.setattr(screen.base.capacity, '_evaluate', lambda *_: metrics)
    observed = []
    original_loss = screen.base.components._batch_loss

    def checked_loss(model, users, positives, negatives, weights):
        assert weights is not None
        observed.extend(weights.detach().cpu().tolist())
        return original_loss(model, users, positives, negatives, weights)

    monkeypatch.setattr(screen.base.components, '_batch_loss', checked_loss)
    model = screen.base._build_model(prep, screen.es.strength_cfg(cfg), screen.spec(), 43)
    root = tmp_path/'arm'
    root.mkdir()
    store = ProgressStore(root/'progress', RunIdentity('test',screen.MODEL_ID,43,'a','b','c'))
    result = screen.es._train(model, prep, cfg, screen.spec(), 43, store, root)
    assert result['selected_epoch'] == 1
    assert result['stopped_epoch'] == 2
    assert observed
    assert not np.allclose(observed, 1)


def test_runner_trains_only_lambda025_and_saves_full_comparison(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(screen.base.v3, 'DEVICE', 'cpu')
    cfg = replace(screen.configure(str(tmp_path)), epochs=2, eval_every=1,
                  patience_start=1, patience=1, batch_size=2)
    monkeypatch.setattr(screen, 'configure', lambda _: cfg)
    prep = inputs()
    prep.update(input_hash='synthetic', source_report='synthetic',
                source_report_sha256=screen.SOURCE_REPORT_SHA,
                screen_config=asdict(cfg))
    _, prep['m4_weights'], prep['m4_diagnostics'] = screen.previous.audit_weights(prep, cfg)
    metrics = {key: 1. for key in (*screen.base.ACCURACY, *screen.es.fixed.PRIMARY)}
    monkeypatch.setattr(screen.base.capacity, '_evaluate', lambda *_: metrics)
    anchors = [dict(model_id=model_id, seed=43, origin='reused', metrics=metrics,
                    selected_epoch=1, stopped_epoch=2,
                    curve=[dict(epoch=1, metrics=metrics)])
               for model_id in ('m1', screen.OLD_MODEL_ID)]
    monkeypatch.setattr(screen, 'verified_anchors', lambda *_: anchors)
    paths = screen.run(cfg, prep)
    result = json.loads(open(paths['json']).read())
    assert [arm['model_id'] for arm in result['arms']] == ['m1', screen.OLD_MODEL_ID, screen.MODEL_ID]
    assert result['arms'][-1]['origin'] == 'trained_lambda025_m4_only'
    assert result['reading']['accuracy_guard_vs_m1']
    assert not result['reading']['both_economic_at10_above_m1']
    assert not result['final_test'] and not result['holdout']
