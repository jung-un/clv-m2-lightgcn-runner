from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

import clv_m5_linear_nv_original_m4_lambda025_screen as screen
from test_linear_nv_original_m4_screen import inputs


def test_only_m4_lambda_changes_and_m5_uses_same_joint_model():
    cfg = screen.configure('unused')
    old = screen.lambda025.previous.configure('unused')
    changed = {key for key in asdict(cfg) if getattr(cfg, key) != getattr(old, key)}
    assert changed == {'positive_weight_lambda'}
    assert cfg.positive_weight_lambda == .25
    assert screen.spec() == dict(screen.lambda025.previous.specs()[1], model_id=screen.MODEL_ID)
    assert screen.spec()['graph'] == 'binary'
    assert screen.spec()['rho'] == .05
    assert screen.spec()['weighted']


def test_missing_exact_lambda025_report_stops_before_data_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(screen.lambda025, 'prepare', lambda *_: pytest.fail('must not load data'))
    with pytest.raises(ValueError, match='Exact completed lambda=.25 report'):
        screen.prepare(tmp_path/'missing.json', tmp_path/'output')


def test_runner_trains_one_joint_m5_and_saves_all_reference_comparisons(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(screen.base.v3, 'DEVICE', 'cpu')
    cfg = replace(screen.configure(str(tmp_path)), epochs=2, eval_every=1,
                  patience_start=1, patience=1, batch_size=2)
    monkeypatch.setattr(screen, 'configure', lambda _: cfg)
    prep = inputs()
    prep['item_economic_valid'][0] = False
    prep['item_bin'][0] = -1
    prep.update(input_hash='synthetic', source_report='synthetic',
                source_report_sha256=screen.SOURCE_REPORT_SHA, screen_config=asdict(cfg))
    _, prep['m4_weights'], prep['m4_diagnostics'] = screen.lambda025.previous.audit_weights(prep, cfg)
    assert prep['m4_diagnostics']['original_invalid_extra_absent']
    metrics = {key: 1. for key in (*screen.base.ACCURACY, *screen.es.fixed.PRIMARY)}
    monkeypatch.setattr(screen.base.capacity, '_evaluate', lambda *_: metrics)
    ids = ('m1', screen.lambda025.OLD_MODEL_ID, screen.OLD_M5, screen.lambda025.MODEL_ID)
    anchors = [dict(model_id=mid, seed=43, origin='reused', metrics=metrics,
                    selected_epoch=1, stopped_epoch=2,
                    curve=[dict(epoch=1, metrics=metrics)]) for mid in ids]
    monkeypatch.setattr(screen, 'verified_anchors', lambda *_: anchors)
    paths = screen.run(cfg, prep)
    report = json.loads(open(paths['json']).read())
    assert [arm['model_id'] for arm in report['arms']] == [*ids, screen.MODEL_ID]
    assert report['arms'][-1]['origin'] == 'trained_lambda025_m5_only'
    assert report['arms'][-1]['selected_epoch'] == 1
    assert report['arms'][-1]['stopped_epoch'] == 2
    assert report['reading']['accuracy_guard_vs_m1']
    assert not report['reading']['both_economic_at10_above_'+screen.lambda025.MODEL_ID]
    assert not report['final_test'] and not report['holdout']
    comparisons = {(row['model_id'], row['reference']) for row in
                   __import__('pandas').read_csv(paths['comparison']).to_dict('records')}
    assert (screen.MODEL_ID, 'm1') in comparisons
    assert (screen.MODEL_ID, screen.lambda025.MODEL_ID) in comparisons
    assert (screen.MODEL_ID, screen.OLD_M5) in comparisons
    assert np.all(np.isfinite(prep['m4_weights']))
