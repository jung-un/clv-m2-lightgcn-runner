"""Read-only price-axis audit for the completed support-backoff M2 checkpoint."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m2_support_backoff_price_screen as screen
from clv_m2_support_backoff_price import PRICE_AXIS
from clv_run_state import file_sha256

VERSION = 'm2-support-backoff-price-selected-checkpoint-diagnostic-v1'
VIEWS = {
    'full': (),
    'without_price': (PRICE_AXIS,),
    'without_user_nv': ('user_n', 'user_v'),
    'id_only': ('user_n', 'user_v', PRICE_AXIS),
}
PAIRS = (
    ('without_price', 'full'),
    ('without_user_nv', 'full'),
    ('id_only', 'full'),
    ('id_only', 'without_price'),
)
KS = (10, 20, 50)


def prepare(reference_report, training_out, diagnostic_out):
    """Rebuild the exact train-only inputs and verify the completed checkpoint."""
    training_out, diagnostic_out = Path(training_out), Path(diagnostic_out)
    if training_out.resolve() == diagnostic_out.resolve():
        raise ValueError('Diagnostic output must differ from the training output')
    report_path = training_out / 'reports/result.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Completed result is missing: {report_path}')
    report = json.loads(report_path.read_text())
    if (report.get('code_version') != screen.VERSION or report.get('final_test') is not False
            or report.get('holdout') is not False or report.get('new_fit_count') != 1):
        raise ValueError('Only the completed development M2 result is in scope')
    cfg, prep = screen.prepare(reference_report, training_out)
    arm = next((a for a in report['arms'] if a['model_id'] == screen.MODEL_ID), None)
    if arm is None or arm['selected_epoch'] != 225 or arm['weighted']:
        raise ValueError('Expected the unweighted M2 selected at epoch225')
    if screen.identity(prep, cfg) != arm['identity']:
        raise ValueError('Prepared data/features differ from the trained M2 identity')
    checkpoint = Path(arm['checkpoint'])
    if not checkpoint.is_file() or file_sha256(checkpoint) != arm['checkpoint_sha256']:
        raise ValueError('Selected checkpoint is missing or changed; no substitute/retraining')
    existing = diagnostic_out / 'diagnostic.json'
    if existing.is_file() and json.loads(existing.read_text()).get('code_version') != VERSION:
        raise ValueError('Diagnostic output belongs to another run')
    print('[진단 준비] 선택225 checkpoint 재사용. 학습·optimizer·epoch 재선택 없음.', flush=True)
    return dict(report=report, arm=arm, cfg=cfg, prepared=prep,
                checkpoint=checkpoint, out_dir=diagnostic_out)


def _load_model(context):
    if file_sha256(context['checkpoint']) != context['arm']['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after preparation')
    model = screen.build(context['prepared'], context['cfg'])
    state = torch.load(context['checkpoint'], map_location='cpu', weights_only=False)
    if state.get('epoch') != context['arm']['selected_epoch'] or 'model_state' not in state:
        raise ValueError('Selected checkpoint schema/epoch mismatch')
    model.load_state_dict(state['model_state'], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _top50(model, prep):
    cache, data = prep['cache'], prep['data']
    users = np.asarray(cache.users, dtype=np.int64)
    user_vectors, item_vectors = model.embeddings()[:2]
    result = np.empty((len(users), 50), dtype=np.int64)
    batch = int(prep['base_cfg']['EVAL_BATCH'])
    for start in range(0, len(users), batch):
        batch_users = users[start:start + batch]
        scores = user_vectors[torch.as_tensor(batch_users, device=user_vectors.device)] @ item_vectors.T
        for row, user in enumerate(batch_users):
            seen = data['csr_items'][data['csr_ptr'][user]:data['csr_ptr'][user + 1]]
            scores[row, seen] = -1e9
        result[start:start + len(batch_users)] = scores.topk(50, dim=1).indices.cpu().numpy()
    return result


@torch.no_grad()
def evaluate_views(model, prep):
    metrics, tops = {}, {}
    try:
        for name, disabled in VIEWS.items():
            model.disabled_axes = disabled
            metrics[name] = screen.base.capacity._evaluate(model, prep)
            tops[name] = _top50(model, prep)
            print(f'[재평가] {name}: {len(prep["cache"].users):,}명 완료 (학습 없음)', flush=True)
    finally:
        model.disabled_axes = ()
    return metrics, tops


def _comparisons(metrics, m1):
    rows = []
    sources = dict(metrics, m1=m1)
    for view, reference in (*PAIRS, *((name, 'm1') for name in VIEWS)):
        if sources[view].keys() != sources[reference].keys():
            raise ValueError(f'Metric sets differ: {view} vs {reference}')
        for metric, value in sources[view].items():
            ref = sources[reference][metric]
            rows.append(dict(view=view, reference=reference, metric=metric, value=value,
                reference_value=ref, delta=value-ref,
                relative_change_pct=100*(value/ref-1) if ref else np.nan))
    return pd.DataFrame(rows)


def _movement_summary(tops, prep):
    cache = prep['cache']
    rows = []
    for view, reference in PAIRS:
        for cutoff in KS:
            for row, user in enumerate(cache.users):
                truth = np.asarray(cache.gt[int(user)])
                weight = np.asarray(cache.rev[int(user)], dtype=float)
                old, new = tops[reference][row, :cutoff], tops[view][row, :cutoff]
                old_hit, new_hit = np.isin(truth, old), np.isin(truth, new)
                gain, loss = new_hit & ~old_hit, old_hit & ~new_hit
                rows.append(dict(view=view, reference=reference, cutoff=cutoff,
                    user=int(user), segment=str(cache.seg[row]),
                    changed=bool(np.any(old != new)), truth_gained=int(gain.sum()),
                    truth_lost=int(loss.sum()), weight_gained=float(weight[gain].sum()),
                    weight_lost=float(weight[loss].sum()),
                    recall_net=float((gain.sum()-loss.sum())/len(truth))))
    per_user = pd.DataFrame(rows)
    summary = []
    for (view, reference, cutoff), frame in per_user.groupby(
            ['view', 'reference', 'cutoff'], sort=False):
        for segment in ('전체', '저CLV', '중CLV', '고CLV'):
            part = frame if segment == '전체' else frame[frame.segment == segment]
            summary.append(dict(view=view, reference=reference, cutoff=cutoff,
                segment=segment, n_users=len(part), changed_user_count=int(part.changed.sum()),
                truth_gained=int(part.truth_gained.sum()), truth_lost=int(part.truth_lost.sum()),
                weighted_gained=float(part.weight_gained.mean()),
                weighted_lost=float(part.weight_lost.mean()),
                weighted_net=float((part.weight_gained-part.weight_lost).mean()),
                recall_net=float(part.recall_net.mean())))
    return per_user, pd.DataFrame(summary)


def run(context):
    model = _load_model(context)
    metrics, tops = evaluate_views(model, context['prepared'])
    expected = context['arm']['metrics']
    for metric, value in expected.items():
        if not np.isclose(metrics['full'][metric], value, rtol=1e-5, atol=1e-8):
            raise RuntimeError(f'Full checkpoint readback differs: {metric}')
    m1 = next(a['metrics'] for a in context['report']['arms'] if a['model_id'] == 'm1')
    comparison = _comparisons(metrics, m1)
    per_user, movement = _movement_summary(tops, context['prepared'])
    absolute = pd.DataFrame([dict(view='m1', **m1)]
        + [dict(view=name, **values) for name, values in metrics.items()])
    primary = ('price_purchase_amount_weighted_hit@10', 'vndcg@10')
    accuracy = tuple(screen.base.ACCURACY)
    reading = {
        'complete': True,
        'new_fit_count': 0,
        'full_readback_matches': True,
        'price_removal_vs_full_pct': {
            key: 100*(metrics['without_price'][key]/metrics['full'][key]-1) for key in primary
        },
        'without_price_accuracy_guard_vs_m1': all(
            metrics['without_price'][key] >= .99*m1[key] for key in accuracy),
        'without_price_both_economic_at10_above_m1': all(
            metrics['without_price'][key] > m1[key] for key in primary),
        'interpretation_limit': (
            'Axis removal is inference-only. id_only retains jointly trained ID parameters and is not M1; '
            'no causal or significance claim.'),
    }
    root = context['out_dir']
    paths = {
        'absolute': root / 'absolute.csv',
        'comparison': root / 'comparison.csv',
        'user_movements': root / 'user_movements.csv',
        'movement_summary': root / 'movement_summary.csv',
        'json': root / 'diagnostic.json',
    }
    for name, frame in [('absolute', absolute), ('comparison', comparison),
                        ('user_movements', per_user), ('movement_summary', movement)]:
        screen.base.capacity.test10._atomic_csv(paths[name], frame)
    screen.base.capacity.test10._atomic_json(paths['json'], dict(
        code_version=VERSION, source_code_version=screen.VERSION,
        selected_epoch=context['arm']['selected_epoch'], checkpoint=str(context['checkpoint']),
        checkpoint_sha256=context['arm']['checkpoint_sha256'], views=VIEWS,
        final_test=False, holdout=False, reading=reading,
        limits='read-only diagnosis on one repeatedly exposed development seed',
        paths={name: str(path) for name, path in paths.items()},
    ))
    print(json.dumps(reading, ensure_ascii=False, indent=2), flush=True)
    return {name: str(path) for name, path in paths.items()}

