"""H&M GraphSAGE execution only: same arithmetic/RNG, prevalidated LOO indices.

No learned tensors cached, no precision/batch/optimizer/epoch changes. Per-epoch
checkpoint uses the original synchronous implementation. CPU integer map is
rebuilt and validated on resume; it is not part of model/checkpoint state.
"""
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from clv_ngcf_training_io import (fixed, save_epoch_local_then_drive,
                                 loss_with_tensor_logs, _sync)


def loss_with_positions(model, users, positives, negatives, weights, positions):
    # Keep the original _pair_scores floating-point operations and ordering.
    user_id, item_id = model._id_embeddings()
    feature, reliability = model._user_feature(users)
    user = model._mix(user_id[users], feature, reliability)
    pos_feature, pos_reliability = model._item_feature(positions, loo=True)
    neg_feature, neg_reliability = model._item_feature(negatives)
    positive = model._mix(item_id[positives], pos_feature, pos_reliability)
    negative = model._mix(item_id[negatives], neg_feature, neg_reliability)
    pos, neg = (user * positive).sum(1), (user * negative).sum(1)
    rows = F.softplus(neg - pos)
    if weights is not None:
        rows = weights * rows
    return rows.mean() + model.batch_l2(users, positives, negatives), (pos > neg).float().mean().detach()


def sample_uniform_negative_matrix(users, positives, n_items, positive_keys, rng, *, k):
    if k != 1:
        raise ValueError('GraphSAGE 고정 uniform K1만 허용합니다')
    negatives = rng.integers(0, n_items, size=len(users))
    offset = users.astype(np.int64) * n_items
    pending = np.arange(len(users))
    for attempt in range(51):
        keys = offset[pending] + negatives[pending]
        positions = np.searchsorted(positive_keys, keys).clip(max=len(positive_keys)-1)
        pending = pending[positive_keys[positions] == keys]
        if not len(pending):
            return negatives[:, None]
        if attempt == 50:
            raise RuntimeError('uniform 음성에 학습 관측상품이 남았습니다')
        negatives[pending] = rng.integers(0, n_items, size=len(pending))


def train(model, prepared, cfg, model_id, seed, store, row_weights=None, *, negative_sampler=None,
          loss_function=None):
    data = prepared['data']
    loss_function = loss_with_tensor_logs if loss_function is None else loss_function
    tr_u, tr_i, positive_keys = data['tr_u'], data['tr_i'], data['pos_key']
    if cfg.negative_count != 1 or len(tr_u) == 0:
        raise ValueError('고정 uniform K1 / 비어 있지 않은 학습자료가 필요합니다')
    if negative_sampler is not None:
        if (positive_keys.ndim != 1 or not len(positive_keys)
                or np.any(positive_keys[1:] <= positive_keys[:-1])):
            raise ValueError('이진검색 음성검사에는 정렬된 고유 학습쌍이 필요합니다')
    else:
        negative_sampler = fixed.m4_helpers.sample_uniform_negative_matrix
    if row_weights is not None:
        row_weights = np.asarray(row_weights)
        if (row_weights.shape != (len(tr_u),) or not np.isfinite(row_weights).all()
                or np.any(row_weights <= 0)):
            raise ValueError('BPR 행 가중치 길이/유한성/양수 조건 불일치')
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    restored = store.restore_epoch(model, optimizer, rng)
    start = 1 if restored is None else restored['next_epoch']
    history = list(restored['history']) if restored else []
    if restored:
        print(f'  [{model_id} s{seed}] epoch {start - 1}에서 재개', flush=True)
    if start > cfg.epochs:
        return history

    positive_positions = None
    if getattr(model, 'm2_active', False):
        keys = model.pair_keys.detach().cpu().numpy()
        if not np.array_equal(keys, prepared['features']['keys']):
            raise ValueError('M2 pair keys differ from prepared features')
        # Construction-only integer cache; no trainable embedding is cached.
        positive_positions = np.empty(len(tr_u), dtype=np.int64)
        for begin in range(0, len(tr_u), 1_000_000):
            end = min(begin + 1_000_000, len(tr_u))
            query = tr_u[begin:end].astype(np.int64) * data['n_items'] + tr_i[begin:end]
            positions = np.searchsorted(keys, query).clip(max=len(keys)-1)
            if not np.array_equal(keys[positions], query):
                raise ValueError('positive가 TRAIN pair에 없습니다')
            positive_positions[begin:end] = positions

    setup = time.perf_counter()
    # Preserve CPU RNG sampling exactly; only immutable training tensors are cached.
    users_all = torch.as_tensor(tr_u, dtype=torch.long, device=device)
    items_all = torch.as_tensor(tr_i, dtype=torch.long, device=device)
    weights_all = (None if row_weights is None else
                   torch.as_tensor(row_weights, dtype=torch.float32, device=device))
    _sync(device)
    setup_seconds = time.perf_counter() - setup
    print(f'[입력 준비] 고정 사용자·상품·가중치 전송 {setup_seconds:.2f}초; 임베딩은 매 배치 재계산', flush=True)
    n_batches = math.ceil(len(tr_u) / cfg.batch_size)
    timing_path = store.root / 'timing' / f'{model_id}_s{seed}.csv'
    timings = pd.read_csv(timing_path).to_dict('records') if timing_path.exists() else []
    timings = [r for r in timings if r['epoch'] < start]
    started = time.perf_counter()
    for epoch in range(start, cfg.epochs + 1):
        epoch_begin = time.perf_counter()
        model.train()
        permutation = rng.permutation(len(tr_u))
        logs, input_seconds, compute_seconds, sampled = [], 0., 0., 0
        for batch in range(n_batches):
            profile = epoch < start + 3 and batch < 5
            if profile:
                _sync(device)
                t0 = time.perf_counter()
            index = permutation[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = negative_sampler(
                users_np, positives_np, data['n_items'], positive_keys, rng, k=1)
            idx = torch.as_tensor(index, dtype=torch.long, device=device)
            users, positives = users_all[idx], items_all[idx]
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)[:, 0]
            weights = None if weights_all is None else weights_all[idx]
            if profile:
                _sync(device)
                t1 = time.perf_counter()
            if positive_positions is not None and loss_function is loss_with_tensor_logs:
                positions = torch.as_tensor(positive_positions[index], dtype=torch.long, device=device)
                loss, correct = loss_with_positions(model, users, positives, negatives, weights, positions)
            else:
                loss, correct = loss_function(model, users, positives, negatives, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs.append(torch.stack((loss.detach(), correct)))
            if profile:
                _sync(device)
                t2 = time.perf_counter()
                input_seconds += t1 - t0
                compute_seconds += t2 - t1
                sampled += 1
                if batch == min(5, n_batches) - 1:
                    print(f'[초반 계측] ep{epoch} / {sampled}배치 평균: 입력·음성추출 {input_seconds/sampled:.3f}초, '
                          f'학습계산 {compute_seconds/sampled:.3f}초 (계측 동기화 포함)', flush=True)
            # Heartbeat retains liveness; no GPU scalar read just to format a log.
            store.heartbeat(epoch=epoch, max_epoch=cfg.epochs, batch=batch + 1,
                            batches=n_batches, selection='none')
        # Convert once, then sum Python floats in the exact original order.
        values = torch.stack(logs).cpu().tolist()
        if not np.isfinite(values).all():
            raise RuntimeError('비유한 학습값: 새 checkpoint/test 평가를 만들지 않습니다')
        history.append({'epoch': epoch, 'loss': sum(v[0] for v in values)/n_batches,
                        'p_correct': sum(v[1] for v in values)/n_batches})
        train_seconds = time.perf_counter() - epoch_begin
        save_begin = time.perf_counter()
        store.save_epoch(model, optimizer, rng, epoch=epoch, history=history,
                         wall_clock_sec=time.perf_counter()-started, selection='none', max_epoch=cfg.epochs)
        save_seconds = time.perf_counter() - save_begin
        timing = {'model_id': model_id, 'seed': seed, 'epoch': epoch,
            'train_seconds': train_seconds, 'save_seconds': save_seconds,
            'profile_batches': sampled,
            'sample_input_seconds_per_batch': input_seconds/sampled if sampled else None,
            'sample_compute_seconds_per_batch': compute_seconds/sampled if sampled else None,
            'fixed_input_setup_seconds': setup_seconds if epoch == start else 0.,
            **getattr(store, 'last_save_timing', {})}
        timings.append(timing)
        fixed.test10._atomic_csv(timing_path, pd.DataFrame(timings))
        if epoch < start + 3 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f'[{model_id} s{seed}] {epoch}/{cfg.epochs} | loss {history[-1]["loss"]:.4f} | '
                  f'P(pos>neg) {history[-1]["p_correct"]:.3f} | 학습 {train_seconds:.1f}초 | '
                  f'Drive 저장완료 {save_seconds:.1f}초', flush=True)
    return history


OPTIMIZATION_VERSION = 'hm-graphsage-execution-v2'

