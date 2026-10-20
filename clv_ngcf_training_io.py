"""NGCF execution-only optimization: unchanged batches, RNG, loss and Adam.

No AMP/CSR/compile, gradient accumulation, embedding cache or async checkpoint.
Timing samples synchronize only the first five batches of the first three
epochs of a session; they are not a speedup benchmark or test evaluation.
"""
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from clv_run_state import clone_state
import lightgcn_clv_m3_dunnhumby_test1 as fixed


def save_epoch_local_then_drive(store, model, optimizer, rng, state):
    """Synchronous publish: retain previous checkpoint until new file is closed.

    Hash the local bytes while copying, never reread the Drive checkpoint just
    for hashing. flush/fsync/replace failures propagate; do not continue training.
    Drive/FUSE service durability itself cannot be guaranteed by this function.
    """
    begin = time.perf_counter()
    payload = {"identity": asdict(store.identity), "model_state": clone_state(model),
        "optimizer_state": optimizer.state_dict(), "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        **state}
    store.latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = store.latest_checkpoint.with_suffix(store.latest_checkpoint.suffix + '.tmp')
    with tempfile.TemporaryDirectory(prefix='clv-ngcf-checkpoint-') as local:
        path = Path(local) / 'checkpoint.pt'
        torch.save(payload, path)
        serialized = time.perf_counter()
        digest = hashlib.sha256()
        with path.open('rb') as source, temporary.open('wb') as target:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        size = temporary.stat().st_size
        if size != path.stat().st_size:
            raise OSError('Drive checkpoint 크기 불일치: 이전 저장본을 보존합니다')
        os.replace(temporary, store.latest_checkpoint)
        published = time.perf_counter()
    store._write_progress('running', {'epoch': int(state['epoch']),
        'max_epoch': state.get('max_epoch', 300), 'selection': 'none',
        'checkpoint_sha256': digest.hexdigest(), 'checkpoint_bytes': size})
    store.last_save_timing = {'checkpoint_local_seconds': serialized - begin,
        'checkpoint_drive_seconds': published - serialized,
        'checkpoint_metadata_seconds': time.perf_counter() - published,
        'checkpoint_bytes': size}
    return store.latest_checkpoint


def loss_with_tensor_logs(model, users, positives, negatives, weights):
    """Same existing _loss math; prevalidated weights, detached tensor logs."""
    positive, negative = model._pair_scores(users, positives, negatives)
    rows = F.softplus(negative - positive)
    if weights is not None:
        rows = weights * rows
    bpr = rows.mean()
    loss = bpr + model.batch_l2(users, positives, negatives)
    correct = (positive > negative).float().mean().detach()
    return loss, correct


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(model, prepared, cfg, model_id, seed, store, row_weights=None, *, negative_sampler=None):
    data = prepared['data']
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
            loss, correct = loss_with_tensor_logs(model, users, positives, negatives, weights)
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
