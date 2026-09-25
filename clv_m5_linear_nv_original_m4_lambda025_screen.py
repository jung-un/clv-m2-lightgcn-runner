"""One-arm M5 screen: linear N/V expression plus validity-masked original M4 at lambda=.25."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_original_m4_lambda025_screen as lambda025
from clv_run_state import ProgressStore, RunIdentity, file_sha256

VERSION = 'm5-linear-nv-original-m4-lambda025-seed43-development-v1'
SOURCE_REPORT_SHA = 'b5de1e97de95296d2f570b46369154c2e423690f7c0a4609660a63d407f62dc1'
MODEL_ID = 'm5_linear_nv_original_validity_masked_lambda025_es'
OLD_M5 = lambda025.previous.M5
es = lambda025.es
base = lambda025.base


def configure(out_dir):
    return lambda025.configure(out_dir)


def spec():
    return dict(lambda025.previous.specs()[1], model_id=MODEL_ID)


def _selected_arm(arm, cfg, prep=None):
    selected = es.replay(arm['curve'], cfg)
    if (arm.get('selected_epoch') != selected['selected_epoch']
            or arm.get('stopped_epoch') != selected['stopped_epoch']
            or arm.get('metrics') != selected['selected_record']['metrics']):
        raise ValueError(f"Source {arm.get('model_id')} selection mismatch")
    if not all(np.isfinite(arm['metrics'].get(k, np.nan))
               for k in (*base.ACCURACY, *es.fixed.PRIMARY)):
        raise ValueError(f"Source {arm.get('model_id')} required metric missing")
    identity = arm.get('identity', {})
    if prep is not None and identity.get('input_hash') != prep['input_hash']:
        raise ValueError(f"Source {arm.get('model_id')} input hash mismatch")
    checkpoint = Path(arm.get('checkpoint', ''))
    if not checkpoint.is_file() or file_sha256(checkpoint) != arm.get('checkpoint_sha256'):
        raise ValueError(f"Source {arm.get('model_id')} checkpoint missing/changed")


def verified_anchors(report_path, cfg, prep=None):
    """Require the exact completed lambda=.25 report and its earlier lambda=.5 anchors."""
    path = Path(report_path)
    if not path.is_file() or file_sha256(path) != SOURCE_REPORT_SHA:
        raise ValueError('Exact completed lambda=.25 report missing/changed; no training started')
    report = json.loads(path.read_text())
    source_cfg = lambda025.configure(report.get('config', {}).get('out_dir', 'source'))
    actual = report.get('config', {})
    expected = json.loads(json.dumps(asdict(source_cfg)))
    proposed = json.loads(json.dumps(asdict(cfg)))
    if (report.get('code_version') != lambda025.VERSION
            or report.get('final_test') is not False or report.get('holdout') is not False
            or report.get('selection') != es.preflight(lambda025.previous.prior.configure(cfg.out_dir))['selection']
            or {k: v for k, v in actual.items() if k != 'reuse_dirs'}
            != {k: v for k, v in expected.items() if k != 'reuse_dirs'}
            or [Path(p).name for p in actual.get('reuse_dirs', [])]
            != [Path(p).name for p in expected['reuse_dirs']]
            or {k: v for k, v in proposed.items() if k != 'out_dir'}
            != {k: v for k, v in expected.items() if k != 'out_dir'}
            or report.get('source_report_sha256') != lambda025.SOURCE_REPORT_SHA):
        raise ValueError('Lambda=.25 source protocol/configuration mismatch')
    prior_path = Path(report['source_report'])
    old_anchors = lambda025.verified_anchors(prior_path, cfg, prep)
    matches = [arm for arm in report.get('arms', [])
               if arm.get('model_id') == lambda025.MODEL_ID and arm.get('seed') == 43]
    if len(matches) != 1:
        raise ValueError('Exactly one completed lambda=.25 M4 arm required')
    m4 = matches[0]
    _selected_arm(m4, cfg, prep)
    if (m4.get('identity', {}).get('version') != lambda025.VERSION
            or m4['identity'].get('baseline_report_sha256') != lambda025.SOURCE_REPORT_SHA
            or m4.get('positive_weight_lambda', cfg.positive_weight_lambda) != .25):
        raise ValueError('Lambda=.25 M4 identity mismatch')
    old_report = json.loads(prior_path.read_text())
    old_matches = [arm for arm in old_report.get('arms', [])
                   if arm.get('model_id') == OLD_M5 and arm.get('seed') == 43]
    if len(old_matches) != 1:
        raise ValueError('Exactly one previous lambda=.5 M5 arm required')
    old_m5 = old_matches[0]
    old_cfg = lambda025.previous.configure(cfg.out_dir)
    _selected_arm(old_m5, old_cfg, prep)
    if old_m5.get('identity', {}).get('version') != lambda025.previous.VERSION:
        raise ValueError('Previous lambda=.5 M5 identity mismatch')
    if prep is not None and report.get('m4_diagnostics') != prep['m4_diagnostics']:
        raise ValueError('Lambda=.25 weight audit differs from source')
    return [old_anchors[0], dict(old_anchors[1], origin='reused_exact_lambda05_report'),
            dict(old_m5, origin='reused_exact_lambda05_report'),
            dict(m4, origin='reused_exact_lambda025_report')]


def prepare(report_path, out_dir):
    cfg = configure(out_dir)
    verified_anchors(report_path, cfg)  # Fail before large data preparation or GPU work.
    report = json.loads(Path(report_path).read_text())
    _, prep, audit = lambda025.prepare(report['source_report'], out_dir)
    anchors = verified_anchors(report_path, cfg, prep)
    prep.update(anchors=anchors, screen_config=asdict(cfg),
                source_report=str(report_path), source_report_sha256=SOURCE_REPORT_SHA)
    print('M1, lambda=.5 M4/M5, lambda=.25 M4 재사용. 새 학습은 lambda=.25 M5 하나만.', flush=True)
    return cfg, prep, audit


def identity(prep, cfg):
    result = lambda025.previous.identity(prep, cfg, spec())
    result.update(version=VERSION, baseline_report_sha256=SOURCE_REPORT_SHA)
    result['source_hashes'][Path(__file__).name] = file_sha256(__file__)
    return result


def save(arms, cfg, prep):
    metrics = {arm['model_id']: arm['metrics'] for arm in arms}
    absolute = pd.DataFrame([dict(model_id=arm['model_id'], seed=43,
        selected_epoch=arm['selected_epoch'], stopped_epoch=arm['stopped_epoch'],
        origin=arm['origin'], **arm['metrics']) for arm in arms])
    comparisons = []
    for model_id, reference in ((lambda025.OLD_MODEL_ID, 'm1'),
                                (lambda025.MODEL_ID, 'm1'),
                                (lambda025.MODEL_ID, lambda025.OLD_MODEL_ID),
                                (OLD_M5, lambda025.OLD_MODEL_ID),
                                (MODEL_ID, 'm1'),
                                (MODEL_ID, lambda025.MODEL_ID),
                                (MODEL_ID, OLD_M5)):
        if model_id not in metrics or reference not in metrics:
            continue
        for key, value in metrics[model_id].items():
            baseline = metrics[reference].get(key)
            if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                comparisons.append(dict(seed=43, model_id=model_id, reference=reference,
                    metric=key, value=value, reference_value=baseline,
                    delta=value-baseline,
                    relative_change_pct=100*(value/baseline-1) if baseline else np.nan))
    comparison = pd.DataFrame(comparisons)
    reading = dict(complete=MODEL_ID in metrics, significance_claim=False,
                   scope='single repeatedly exposed development seed; full metrics and unfavorable outcomes retained')
    if MODEL_ID in metrics:
        reading['accuracy_guard_vs_m1'] = all(metrics[MODEL_ID][k] >= .99*metrics['m1'][k]
                                               for k in base.ACCURACY)
        for reference in ('m1', lambda025.MODEL_ID, OLD_M5):
            reading['both_economic_at10_above_'+reference] = all(
                metrics[MODEL_ID][k] > metrics[reference][k] for k in es.fixed.PRIMARY)
    curve = pd.DataFrame([dict(model_id=arm['model_id'], epoch=row['epoch'], **row['metrics'])
                          for arm in arms for row in arm['curve']])
    root = Path(cfg.out_dir)/'reports'
    paths = {}
    for name, frame in (('absolute', absolute), ('comparison', comparison), ('curve', curve)):
        paths[name] = str(root/f'{name}.csv')
        base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['json'] = str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']), dict(
        code_version=VERSION, config=asdict(cfg),
        selection=es.preflight(lambda025.previous.prior.configure(cfg.out_dir))['selection'],
        reading=reading, arms=arms, final_test=False, holdout=False,
        source_report=prep['source_report'], source_report_sha256=SOURCE_REPORT_SHA,
        m4_diagnostics=prep['m4_diagnostics'], paths=paths))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    anchors = verified_anchors(prep['source_report'], cfg, prep)
    _, weights, diagnostic = lambda025.previous.audit_weights(prep, cfg)
    if not diagnostic['original_invalid_extra_absent'] or not np.array_equal(weights, prep['m4_weights']):
        raise ValueError('Prepared M4 weights changed or invalid-input weighting recurred')
    arm_spec = spec()
    ident = identity(prep, cfg)
    key = base._digest(ident)
    root = Path(cfg.out_dir)/'arms'/key
    path = root/'result.json'
    if path.is_file():
        arm = json.loads(path.read_text())
        selected = es.replay(arm['curve'], cfg)
        if (arm['identity'] != ident or arm['metrics'] != selected['selected_record']['metrics']
                or arm['selected_epoch'] != selected['selected_epoch']
                or arm['stopped_epoch'] != selected['stopped_epoch']):
            raise ValueError('Cached M5 result mismatch')
    else:
        store = ProgressStore(root/'progress', RunIdentity(VERSION, MODEL_ID, 43, key,
            'content:'+base._digest(ident['source_hashes']), prep['input_hash']))
        model = es.fixed._build(prep, es.strength_cfg(cfg), arm_spec, 43)
        result = es._train(model, prep, cfg, arm_spec, 43, store, root)
        arm = dict(**arm_spec, seed=43, identity=ident,
                   origin='trained_lambda025_m5_only', **result)
        base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path),
            checkpoint_path=result['checkpoint'])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return save(anchors+[arm], cfg, prep)
