"""Read-only, selected-epoch N/V masking and input-shift audit; never fit a model."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

import clv_nv_modulated_l2_screen as source
import clv_m5_nv_score_ablation_diagnostic as shared
from clv_run_state import file_sha256

VERSION = 'nv-modulated-m2-selected-checkpoint-diagnostic-v1'
REPORT_SHA = 'd9b1696e95800ff2251b652f20ecaa8b895d8095f063eb77a58aa8d81d0650c2'
VIEWS = ('id_only', 'id_n', 'id_v', 'full')
PAIRS = (('id_n', 'id_only'), ('id_v', 'id_only'), ('full', 'id_only'),
         ('full', 'id_n'), ('full', 'id_v'))
KS = (10, 20, 50)


def preflight(report_path, checkpoint_path=None):
    """Verify exact source before loading data. A moved checkpoint needs the same hash."""
    path = Path(report_path)
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            payload = archive.read('result.json')
    else:
        payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA:
        raise ValueError('최신 M2 L2=.0001 완료 result.json/ZIP과 다릅니다. 학습하지 않습니다.')
    report = json.loads(payload)
    cfg = source.configure(report['config']['out_dir'])
    expected_config = json.loads(json.dumps(asdict(cfg)))
    actual_config = dict(report['config'])
    # Reuse paths differ between local and Colab; never relax numerical settings.
    for config in (expected_config, actual_config):
        config['reuse_dirs'] = [Path(p).name for p in config['reuse_dirs']]
    if (report['code_version'] != source.VERSION or report['final_test'] is not False
            or report['holdout'] is not False or report['feature_settings'] != source.SETTINGS
            or actual_config != expected_config
            or report['selection'] != source.selection(cfg)):
        raise ValueError('Source configuration/selection mismatch')
    cfg = replace(cfg, reuse_dirs=tuple(report['config']['reuse_dirs']))
    arm = shared.movement._one_arm(report, source.MODEL_ID)
    source.check_selected(arm, cfg)
    if arm['selected_epoch'] != 100 or arm['weighted']:
        raise ValueError('Only the selected, unweighted M2 epoch100 is in scope')
    for name, digest in arm['identity']['source_hashes'].items():
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'학습 당시 코드와 다릅니다: {name}')
    checkpoint = Path(checkpoint_path or arm['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(f'선택 checkpoint가 없습니다: {checkpoint}. 자동 재학습 없음.')
    if file_sha256(checkpoint) != arm['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch; do not substitute epoch200/latest')
    return report, cfg, arm, checkpoint


def prepare(report_path, out_dir, checkpoint_path=None):
    report, cfg, arm, checkpoint = preflight(report_path, checkpoint_path)
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path(report['config']['out_dir']).resolve():
        raise ValueError('Use a separate diagnostic output directory, not the training root')
    existing = out_dir/'diagnostic.json'
    if existing.is_file() and json.loads(existing.read_text()).get('code_version') != VERSION:
        raise ValueError('Output belongs to another diagnostic')
    # Reuse only preparation, never source.prepare/run/_train or a baseline-fitting path.
    prep_cfg = source.configure(str(out_dir))
    prep = source.base._prepare(source.es.strength_cfg(prep_cfg))
    d = prep['data']
    if (set(d['splits']) != {'test'} or float(d['train'].t.max()) != 683.
            or prep['base_cfg'].get('EVAL_HOLDOUT') or prep['base_cfg']['MIN_ITEM_INTER'] != 1):
        raise ValueError('Only exposed development days684–690 and full catalog are allowed')
    shared.movement._verify_new_item_truth(prep)
    f = source.previous.build_features(d['train'], n_users=d['n_users'], n_items=d['n_items'],
        q_n=prep['q_n'], q_v=prep['q_v'], valid=prep['clv_valid'],
        shrinkage=cfg.shrinkage_strength, bandwidth=cfg.basis_bandwidth)
    if not np.array_equal(f['keys'], d['pos_key']):
        raise ValueError('Feature/graph pair mismatch')
    prep.update(features=f, features_sha256=source.previous.feature_hash(f),
                feature_settings=dict(source.SETTINGS), source_report_sha256=source.SOURCE_SHA)
    if source.identity(prep, cfg) != arm['identity']:
        raise ValueError('Prepared input/features/source identity differs from trained M2')
    print('[진단 준비] 최신 M2 epoch100 한 개. 학습·optimizer·epoch 재선택 없음.', flush=True)
    return dict(report=report, cfg=cfg, arm=arm, checkpoint=checkpoint,
                prepared=prep, out_dir=out_dir, report_path=str(report_path))


def load_model(context):
    path, arm = context['checkpoint'], context['arm']
    if file_sha256(path) != arm['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after preparation')
    model = source.previous._build(context['prepared'], context['cfg'])
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('epoch') != 100 or set(state) != {'epoch', 'model_state'}:
        raise ValueError('Selected checkpoint schema/epoch mismatch')
    model.load_state_dict(state['model_state'], strict=True)
    model.eval()
    return model


@torch.no_grad()
def view_embeddings(model):
    uid, iid = model._id_embeddings()
    factors = {name: model._modulation(name, getattr(model, name+'_input'))
               for name in model.encoders}
    views = {}
    for name, axes in (('id_only', ()), ('id_n', ('n',)), ('id_v', ('v',)), ('full', ('n', 'v'))):
        views[name] = tuple(z*(1+sum(factors[side+'_'+a] for a in axes))
                            for side, z in (('user', uid), ('item', iid)))
    original = model.embeddings()[:2]
    for computed, actual in zip(views['full'], original):
        torch.testing.assert_close(computed, actual, rtol=1e-5, atol=2e-6)
    views['full'] = original
    return views


@torch.no_grad()
def rank_views(views, prep):
    """Same unseen candidates for every view; never create a repeat-purchase task."""
    cache, d = prep['cache'], prep['data']
    users = np.asarray(cache.users, dtype=np.int64)
    if d['n_items'] < 50 or len(np.unique(users)) != len(users):
        raise ValueError('Top50 needs >=50 items and unique evaluation users')
    tops = {}
    batch = int(prep['base_cfg']['EVAL_BATCH'])
    if batch <= 0:
        raise ValueError('Positive evaluation batch required')
    for name in VIEWS:
        uv, iv = views[name]
        if not torch.isfinite(uv).all() or not torch.isfinite(iv).all():
            raise ValueError('Nonfinite checkpoint embeddings')
        top = np.empty((len(users), 50), dtype=np.int64)
        for start in range(0, len(users), batch):
            bu = users[start:start+batch]
            scores = uv[torch.as_tensor(bu, device=uv.device)] @ iv.T
            for row, u in enumerate(bu):
                seen = d['csr_items'][d['csr_ptr'][u]:d['csr_ptr'][u+1]]
                if d['n_items']-len(seen) < 50:
                    raise ValueError('Fewer than50 unseen candidates for an evaluation user')
                scores[row, seen] = -1e9
            top[start:start+len(bu)] = scores.topk(50, dim=1).indices.cpu().numpy()
        tops[name] = top
        print(f'[재평가] {name}: {len(users):,}명 완료 (학습 없음)', flush=True)
    return tops


def metric_readback(absolute, expected):
    """Reconcile every original overall/segment/exposure metric, not only @10."""
    measured = {}
    for r in absolute[absolute.component == 'full'].itertuples():
        key = r.metric
        if r.segment == '전체' and key == 'arp':
            key = 'mean_recommended_price_percentile'
        if r.segment == '전체' and key == 'entropy':
            key = 'exposure_entropy'
        if r.segment != '전체':
            key = r.segment+'_'+('revenue' if key == 'price_purchase_amount_weighted_hit' else key)
        if r.metric != 'user_value_tendency_recommended_price_alignment':
            key += f'@{r.cutoff}'
        measured[key] = float(r.value)
    checks = []
    for key, value in expected.items():
        if key not in measured or not np.isclose(measured[key], value, rtol=1e-5, atol=1e-7):
            raise ValueError(f'Original metric not reproduced: {key}: {measured.get(key)} vs {value}')
        checks.append(dict(metric=key, expected=value, measured=measured[key],
                           absolute_difference=abs(measured[key]-value)))
    return pd.DataFrame(checks)


def comparisons(absolute):
    p = absolute.pivot(index=['segment', 'cutoff', 'metric'], columns='component', values='value')
    frames = []
    for name, reference in PAIRS:
        f = p[[name, reference]].rename(columns={name: 'value', reference: 'reference_value'}).reset_index()
        f['view'], f['reference'] = name, reference
        f['delta'] = f.value-f.reference_value
        f['relative_change_pct'] = 100*f.delta/f.reference_value.replace(0, np.nan)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def movements(tops, prep):
    """Keep truth ranks once per pair; summarize gains/losses at all three cutoffs."""
    cache = prep['cache']
    truth_frames, user_rows, summaries = [], [], []
    degree = np.bincount(prep['features']['keys'] % prep['data']['n_items'],
                         minlength=prep['data']['n_items'])
    for name, reference in PAIRS:
        truth, _ = shared.movement.movement_tables(cache.users, cache.seg, tops[reference],
                                                   tops[name], cache.gt, cache.rev)
        truth = truth.rename(columns={'m4_rank': 'reference_rank', 'm5_rank': 'view_rank',
                                      'm4_bucket': 'reference_bucket', 'm5_bucket': 'view_bucket'})
        truth['view'], truth['reference'] = name, reference
        truth['item_training_buyers'] = degree[truth.item.to_numpy(np.int64)]
        truth_frames.append(truth)
        for k in KS:
            for row, u in enumerate(cache.users):
                actual = np.asarray(cache.gt[int(u)])
                weight = np.asarray(cache.rev[int(u)], dtype=float)
                before, after = tops[reference][row, :k], tops[name][row, :k]
                old_hit, new_hit = np.isin(actual, before), np.isin(actual, after)
                gain, loss = new_hit & ~old_hit, old_hit & ~new_hit
                user_rows.append(dict(view=name, reference=reference, cutoff=k, user=int(u),
                    segment=str(cache.seg[row]), truth_count=len(actual),
                    candidate_entries=len(set(after)-set(before)), truth_gained=int(gain.sum()),
                    truth_lost=int(loss.sum()), weight_gained=float(weight[gain].sum()),
                    weight_lost=float(weight[loss].sum()), recall_net=float((gain.sum()-loss.sum())/len(actual))))
    per_user = pd.DataFrame(user_rows)
    for (name, reference, k), frame in per_user.groupby(['view', 'reference', 'cutoff'], sort=False):
        for segment in ('전체', '저CLV', '중CLV', '고CLV'):
            sub = frame if segment == '전체' else frame[frame.segment == segment]
            if sub.empty:
                continue
            summaries.append(dict(view=name, reference=reference, cutoff=k, segment=segment,
                n_users=len(sub), changed_user_count=int((sub.candidate_entries > 0).sum()),
                truth_gained=int(sub.truth_gained.sum()), truth_lost=int(sub.truth_lost.sum()),
                weighted_gained=float(sub.weight_gained.mean()), weighted_lost=float(sub.weight_lost.mean()),
                weighted_net=float((sub.weight_gained-sub.weight_lost).mean()), recall_net=float(sub.recall_net.mean())))
    return pd.concat(truth_frames, ignore_index=True), per_user, pd.DataFrame(summaries)


@torch.no_grad()
def input_shift(model, prep, views):
    """Matched training pairs only. Full-positive scoring is diagnostic, not a fix."""
    d, f = prep['data'], prep['features']
    rng = np.random.default_rng(4301)
    ix = rng.choice(len(d['tr_u']), min(8192, len(d['tr_u'])), replace=False)
    u, i = d['tr_u'][ix], d['tr_i'][ix]
    positions = np.searchsorted(f['keys'], u*d['n_items']+i)
    if not np.array_equal(f['keys'][positions], u*d['n_items']+i):
        raise ValueError('Training probe pair missing')
    negatives = source.base.components.m4_helpers.sample_uniform_negative_matrix(
        u, i, d['n_items'], d['pos_key'], rng, k=1).reshape(-1)
    device = next(model.parameters()).device
    ut, it, nt, pt = [torch.as_tensor(a, dtype=torch.long, device=device)
                      for a in (u, i, negatives, positions)]
    _, iid = views['id_only']
    uu = views['full'][0][ut]
    loo_item = iid[it]*(1+sum(model._modulation('item_'+axis, getattr(model, 'loo_'+axis+'_input')[pt])
                            for axis in ('n', 'v')))
    pos_loo = (uu*loo_item).sum(1)
    pos_full = (uu*views['full'][1][it]).sum(1)
    neg_full = (uu*views['full'][1][nt]).sum(1)
    pair_pos, pair_neg = model._pair_scores(ut, it, nt)
    torch.testing.assert_close(pos_loo, pair_pos, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(neg_full, pair_neg, atol=2e-5, rtol=1e-5)
    degree = np.bincount(f['keys'] % d['n_items'], minlength=d['n_items'])
    rows = dict(user=u, item=i, negative_item=negatives, item_buyers=degree[i],
                positive_loo_score=pos_loo.cpu().numpy(), positive_full_score=pos_full.cpu().numpy(),
                negative_full_score=neg_full.cpu().numpy())
    for axis in ('n', 'v'):
        full, loo = f['item_'+axis][i], f['loo_'+axis][positions]
        rows[axis+'_full_valid'] = np.any(full != 0, axis=1)
        rows[axis+'_loo_valid'] = np.any(loo != 0, axis=1)
        rows[axis+'_input_l2_difference'] = np.linalg.norm(full-loo, axis=1)
    frame = pd.DataFrame(rows)
    frame['full_minus_loo_score'] = frame.positive_full_score-frame.positive_loo_score
    frame['loo_pair_bpr'] = np.logaddexp(0, frame.negative_full_score-frame.positive_loo_score)
    frame['full_pair_bpr'] = np.logaddexp(0, frame.negative_full_score-frame.positive_full_score)
    frame['buyer_group'] = np.select([frame.item_buyers == 1, frame.item_buyers <= 5],
                                     ['1', '2-5'], default='6+')
    summary = []
    for group in ('all', '1', '2-5', '6+'):
        sub = frame if group == 'all' else frame[frame.buyer_group == group]
        if sub.empty:
            continue
        summary.append(dict(buyer_group=group, n_train_rows=len(sub),
            mean_signed_score_shift=float(sub.full_minus_loo_score.mean()),
            mean_abs_score_shift=float(sub.full_minus_loo_score.abs().mean()),
            mean_loo_bpr=float(sub.loo_pair_bpr.mean()), mean_full_bpr=float(sub.full_pair_bpr.mean()),
            **{axis+'_full_valid_loo_missing_share': float((sub[axis+'_full_valid'] & ~sub[axis+'_loo_valid']).mean())
               for axis in ('n', 'v')}))
    return frame, pd.DataFrame(summary)


def run(context):
    started = time.monotonic()
    model = load_model(context)
    prep = context['prepared']
    views = view_embeddings(model)
    tops = rank_views(views, prep)
    absolute = shared._metric_rows(prep, tops)
    readback = metric_readback(absolute, context['arm']['metrics'])
    comparison = comparisons(absolute)
    truth, users, summary = movements(tops, prep)
    lookup = comparison.set_index(['view', 'reference', 'segment', 'cutoff', 'metric'])
    for row in summary.itertuples():
        for measure, observed in (('price_purchase_amount_weighted_hit', row.weighted_net),
                                   ('recall', row.recall_net)):
            expected = lookup.loc[(row.view, row.reference, row.segment, row.cutoff, measure), 'delta']
            if not np.isclose(observed, expected, atol=1e-7, rtol=1e-5):
                raise ValueError('Truth movements do not reconcile to metrics')
    shift, shift_summary = input_shift(model, prep, views)
    # Save unambiguous diagnostic views, not new model arms or selected operating points.
    frames = dict(absolute=absolute, comparison=comparison, metric_readback=readback,
                  truth_movements=truth, user_movements=users, movement_summary=summary,
                  train_input_shift=shift, train_input_shift_summary=shift_summary)
    root = context['out_dir']
    paths = {name: str(root/f'{name}.csv') for name in frames}
    for name, frame in frames.items():
        source.base.capacity.test10._atomic_csv(Path(paths[name]), frame)
    paths['diagnostic'] = str(root/'diagnostic.json')
    report = dict(code_version=VERSION, model_id=source.MODEL_ID, seed=43, selected_epoch=100,
        source_result=context['report_path'], source_result_sha256=REPORT_SHA,
        checkpoint=str(context['checkpoint']), checkpoint_sha256=context['arm']['checkpoint_sha256'],
        input_hash=prep['input_hash'], features_sha256=prep['features_sha256'],
        source_hashes={name: file_sha256(Path(__file__).with_name(name)) for name in
                      (Path(__file__).name, 'clv_m5_nv_score_ablation_diagnostic.py',
                       'clv_m4_m5_top10_movement_diagnostic.py')},
        original_metric_count=len(readback), views=VIEWS, comparisons=PAIRS,
        split='historical_development_days_684_690', new_training=False, optimizer_updates=0,
        checkpoint_selection_changed=False, final_test=False, holdout=False,
        significance_claim=False, causal_attribution_claim=False,
        view_note='same jointly trained M2; axis masking at inference only; ID-only is not M1; masked views are not proposed models',
        cross_note='full is multiplicative and includes N/V cross terms; effects need not add',
        input_shift_note='fixed TRAIN sample of <=8192 rows, seed4301; matched positive LOO/full audit only, not unseen-item performance or causal proof; never remove LOO as a fix',
        m1_user_level_available=False, m1_aggregate_reference=shared.movement._one_arm(context['report'], 'm1')['metrics'],
        m1_note='original matched M1 has no selected checkpoint; no refit, no substitution by M2 ID-only',
        elapsed_seconds=time.monotonic()-started, paths=paths)
    source.base.capacity.test10._atomic_json(Path(paths['diagnostic']), report)
    print('[완료] 원본 전체 지표 재현·정답 이동 합계 대조 통과. 추가 학습0회.', flush=True)
    return paths
