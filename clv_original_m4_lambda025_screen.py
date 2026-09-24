"""One-factor Dunnhumby screen: validity-masked original M4, lambda 0.25."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_linear_nv_original_m4_screen as previous
from clv_run_state import ProgressStore, RunIdentity, file_sha256

VERSION = 'original-m4-validity-masked-lambda025-seed43-development-v1'
SOURCE_REPORT_SHA = '7bf4632dc2b1f53b324cac320ddf107856fa34d4f64d0cfd43bc1b34bacb9dce'
MODEL_ID = 'm4_original_validity_masked_lambda025_es'
OLD_MODEL_ID = previous.M4
LAMBDA = 0.25

es = previous.es
base = previous.base


def configure(out_dir):
    """Change only the original M4 positive-row weight strength."""
    return replace(previous.configure(str(out_dir)), positive_weight_lambda=LAMBDA)


def spec():
    return dict(previous.specs()[0], model_id=MODEL_ID)


def verified_anchors(report_path, cfg, prep=None):
    """Reuse exactly the supplied seed-43 M1 and lambda-0.5 original M4."""
    path = Path(report_path)
    if not path.is_file() or file_sha256(path) != SOURCE_REPORT_SHA:
        raise ValueError('Exact completed lambda-0.5 report missing/changed; no training started')
    report = json.loads(path.read_text())
    old_cfg = previous.configure(report.get('config', {}).get('out_dir', 'source'))
    expected = json.loads(json.dumps(asdict(old_cfg)))
    if (report.get('code_version') != previous.VERSION or report.get('final_test') is not False
            or report.get('holdout') is not False or report.get('selection') != es.preflight(
                previous.prior.configure(old_cfg.out_dir))['selection']):
        raise ValueError('Source report protocol mismatch')
    # The default Drive root differs from a local smoke-test root. Reuse
    # directories are not consulted by this runner; compare their identities.
    source_config = report['config']
    if ({k: v for k, v in source_config.items() if k != 'reuse_dirs'}
            != {k: v for k, v in expected.items() if k != 'reuse_dirs'}
            or [Path(p).name for p in source_config['reuse_dirs']]
            != [Path(p).name for p in expected['reuse_dirs']]):
        raise ValueError('Source report configuration mismatch')
    changed = {k for k in asdict(cfg) if k != 'reuse_dirs'
               and getattr(cfg, k) != getattr(old_cfg, k)}
    if (changed != {'out_dir', 'positive_weight_lambda'}
            or [Path(p).name for p in cfg.reuse_dirs]
            != [Path(p).name for p in old_cfg.reuse_dirs]
            or cfg.positive_weight_lambda != LAMBDA):
        raise ValueError('Only out_dir and M4 lambda may differ from the source')
    arms = []
    for model_id in ('m1', OLD_MODEL_ID):
        matches = [arm for arm in report['arms']
                   if arm.get('model_id') == model_id and arm.get('seed') == 43]
        if len(matches) != 1:
            raise ValueError(f'Exactly one seed-43 {model_id} required')
        arm = matches[0]
        selected = es.replay(arm['curve'], old_cfg)
        if (arm['selected_epoch'] != selected['selected_epoch']
                or arm['stopped_epoch'] != selected['stopped_epoch']
                or arm['metrics'] != selected['selected_record']['metrics']):
            raise ValueError(f'Source {model_id} selection mismatch')
        if not all(np.isfinite(arm['metrics'].get(k, np.nan))
                   for k in (*base.ACCURACY, *es.fixed.PRIMARY)):
            raise ValueError(f'Source {model_id} required metric missing')
        if model_id == 'm1':
            source = Path(arm.get('source_result', ''))
            if not source.is_file() or file_sha256(source) != arm.get('source_sha256'):
                raise ValueError('M1 source result missing/changed; no fallback training')
        else:
            identity = arm['identity']
            if (identity.get('version') != previous.VERSION
                    or identity.get('baseline_report_sha256') != previous.prior.EXPECTED_REPORT_SHA):
                raise ValueError('Original M4 source identity mismatch')
            checkpoint = Path(arm.get('checkpoint', ''))
            if not checkpoint.is_file() or file_sha256(checkpoint) != arm.get('checkpoint_sha256'):
                raise ValueError('Original M4 checkpoint missing/changed')
            if prep is not None and identity.get('input_hash') != prep['input_hash']:
                raise ValueError('Input hash differs from matched baselines')
        arms.append(dict(arm, origin='reused_exact_lambda05_report'))
    return arms


def prepare(report_path, out_dir):
    cfg = configure(out_dir)
    verified_anchors(report_path, cfg)  # Fail before any large data preparation.
    prep = base._prepare(es.strength_cfg(previous.configure(cfg.out_dir)))
    anchors = verified_anchors(report_path, cfg, prep)
    audit, weights, diagnostic = previous.audit_weights(prep, cfg)
    if not diagnostic['original_invalid_extra_absent']:
        raise RuntimeError('Invalid-input rows received extra weight; no training started')
    root = Path(cfg.out_dir)
    base.capacity.test10._atomic_csv(root/'m4_validity_audit_lambda025.csv', audit)
    base.capacity.test10._atomic_json(root/'m4_validity_audit_lambda025.json', diagnostic)
    prep.update(anchors=anchors, m4_weights=weights, m4_diagnostics=diagnostic,
                screen_config=asdict(cfg), source_report=str(report_path),
                source_report_sha256=SOURCE_REPORT_SHA)
    print('M1 및 lambda=0.5 수정 원형 M4 재사용; 새 학습은 lambda=0.25 M4 한 개만.', flush=True)
    return cfg, prep, audit


def identity(prep, cfg):
    result = previous.identity(prep, cfg, spec())
    result.update(version=VERSION, baseline_report_sha256=SOURCE_REPORT_SHA)
    result['source_hashes'][Path(__file__).name] = file_sha256(__file__)
    return result


def save(arms, cfg, prep):
    metrics = {arm['model_id']: arm['metrics'] for arm in arms}
    absolute = pd.DataFrame([dict(model_id=arm['model_id'], seed=43,
        selected_epoch=arm['selected_epoch'], stopped_epoch=arm['stopped_epoch'],
        origin=arm['origin'], **arm['metrics']) for arm in arms])
    comparisons = []
    for model_id, reference in ((OLD_MODEL_ID, 'm1'), (MODEL_ID, 'm1'), (MODEL_ID, OLD_MODEL_ID)):
        if model_id not in metrics:
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
                   scope='one repeatedly exposed development seed; @20/@50, all segments and exposure reported')
    if MODEL_ID in metrics:
        reading['accuracy_guard_vs_m1'] = all(metrics[MODEL_ID][k] >= .99*metrics['m1'][k]
                                               for k in base.ACCURACY)
        for reference in ('m1', OLD_MODEL_ID):
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
        selection=es.preflight(previous.prior.configure(cfg.out_dir))['selection'],
        reading=reading, arms=arms, final_test=False, holdout=False,
        source_report=prep['source_report'], source_report_sha256=SOURCE_REPORT_SHA,
        m4_diagnostics=prep['m4_diagnostics'], paths=paths))
    return paths


def run(cfg, prep):
    if cfg != configure(cfg.out_dir) or asdict(cfg) != prep['screen_config']:
        raise ValueError('Configuration changed after prepare')
    anchors = verified_anchors(prep['source_report'], cfg, prep)
    _, weights, diagnostic = previous.audit_weights(prep, cfg)
    if not diagnostic['original_invalid_extra_absent'] or not np.array_equal(weights, prep['m4_weights']):
        raise ValueError('Prepared weights changed or invalid-input weighting recurred')
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
            raise ValueError('Cached result mismatch')
    else:
        store = ProgressStore(root/'progress', RunIdentity(VERSION, MODEL_ID, 43, key,
            'content:'+base._digest(ident['source_hashes']), prep['input_hash']))
        model = base._build_model(prep, es.strength_cfg(cfg), arm_spec, 43)
        result = es._train(model, prep, cfg, arm_spec, 43, store, root)
        arm = dict(**arm_spec, seed=43, identity=ident,
                   origin='trained_lambda025_m4_only', **result)
        base.capacity.test10._atomic_json(path, arm)
        store.mark_complete(epoch=result['stopped_epoch'], max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'], result_path=str(path),
            checkpoint_path=result['checkpoint'])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return save(anchors+[arm], cfg, prep)
