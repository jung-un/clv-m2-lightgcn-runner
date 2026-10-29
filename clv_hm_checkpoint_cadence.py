"""Opt-in storage-only wrapper; install AFTER quota and retention wrappers."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import time
import torch


def install(screen, prepared, *, local_root='/content/hm_epoch_checkpoints', every=5):
    cls = getattr(screen, 'original', screen).SafeProgressStore
    if every != 5:
        raise ValueError('Approved Drive interval is five epochs')
    if not getattr(cls, '_retention_installed', False):
        raise RuntimeError('Install latest/previous retention first')
    root = Path(prepared['run_dir']).resolve() / 'progress'
    if getattr(cls, '_hm_cadence_root', None) == str(root):
        return
    if getattr(cls, '_hm_cadence_root', None):
        raise RuntimeError('Another run already installed')
    save_drive, restore_drive = cls.save_epoch, cls.restore_epoch

    def paths(store):
        if store.root.resolve() != root:
            raise RuntimeError('Checkpoint run path mismatch')
        key = hashlib.sha256(json.dumps(asdict(store.identity), sort_keys=True).encode()).hexdigest()
        directory = Path(local_root) / key
        directory.mkdir(parents=True, exist_ok=True)
        return directory / 'latest.pt', directory / 'previous.pt'

    def read(store, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        store._validate_identity(payload['identity'])
        epoch = payload['epoch']
        if not isinstance(epoch, int) or not 1 <= epoch <= 300:
            raise RuntimeError('Invalid checkpoint epoch')
        if [r['epoch'] for r in payload['history']] != list(range(1, epoch + 1)):
            raise RuntimeError('Invalid checkpoint history')
        return payload

    def save(self, model, optimizer, rng, **state):
        latest, previous = paths(self)
        begin = time.perf_counter()
        payload = dict(identity=asdict(self.identity), model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(), numpy_rng_state=rng.bit_generator.state,
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], **state)
        temporary = latest.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if latest.exists():
            os.replace(latest, previous)
        os.replace(temporary, latest)
        due = state['epoch'] % every == 0 or state['epoch'] == state.get('max_epoch', 300)
        if due:
            result = save_drive(self, model, optimizer, rng, **state)
            self.last_save_timing = {**getattr(self, 'last_save_timing', {}), 'drive_published': True}
            print(f"[저장 주기] epoch {state['epoch']}: 로컬 + Drive 저장·보존 검증 완료", flush=True)
            return result
        self.last_save_timing = dict(checkpoint_local_seconds=time.perf_counter()-begin,
            checkpoint_drive_seconds=0., checkpoint_metadata_seconds=0.,
            checkpoint_bytes=latest.stat().st_size, drive_published=False)
        return latest

    def restore(self, model, optimizer, rng):
        latest, previous = paths(self)
        drive_path = self.latest_checkpoint
        drive_epoch = read(self, drive_path)['epoch'] if drive_path.exists() else 0
        candidates = [(read(self, p)['epoch'], p) for p in (latest, previous) if p.exists()]
        epoch, path = max(candidates, default=(0, latest))
        if epoch <= drive_epoch:
            return restore_drive(self, model, optimizer, rng)
        payload = read(self, path)
        self.latest_checkpoint = path
        try:
            restored = restore_drive(self, model, optimizer, rng)
        finally:
            self.latest_checkpoint = drive_path
        # A failed final Drive publish must be retried before final evaluation.
        if epoch == 300:
            state = {k: payload[k] for k in ('epoch', 'history', 'wall_clock_sec', 'selection', 'max_epoch') if k in payload}
            save_drive(self, model, optimizer, rng, **state)
        print(f'[로컬 재개] epoch {epoch}; Drive 마지막 저장 {drive_epoch}', flush=True)
        return restored

    cls.save_epoch, cls.restore_epoch = save, restore
    cls._hm_cadence_root = str(root)
    print('로컬 매epoch / Drive 5epoch 및 최종epoch 저장. 기존 Drive 최신·직전 보존/용량 가드 유지.', flush=True)
