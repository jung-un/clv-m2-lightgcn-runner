"""Reuse verified, untrained CPU preprocessing bytes across different CPUs.

No model/optimizer/checkpoint is transferred. Original reference and feature
hash gates remain active; the source L4 must pass its original preparation.
"""
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import launch as platform


def read_snapshot(path, expected_sha256):
    if platform.sha(path) != expected_sha256:
        raise RuntimeError('L4 고정 입력 파일 해시 불일치')
    import torch
    # torch.load seeks repeatedly; decompress once onto the local container disk.
    with tempfile.TemporaryDirectory(prefix='dh-gat-inputs-') as folder:
        local = Path(folder)/'inputs.pt'
        with gzip.open(path,'rb') as source, local.open('wb') as output:
            shutil.copyfileobj(source,output,8*1024**2)
        return torch.load(local,map_location='cpu',weights_only=False)


def export():
    import torch
    import dh_gat_partition as partition
    if os.environ.get('RUNPOD_POD_ID') != 'w05yjhfwulnyr4':
        raise RuntimeError('검증된 L4에서만 전처리 입력을 고정합니다')
    screen, cfg = platform.load_runner('dh-gat')
    original_cfg = screen.original.configure(
        out_dir=str(partition.CONTROL/'snapshot_prepare'),
        m2_reference=cfg.m2_reference, m4_reference=cfg.m4_reference)
    prepared = screen.original.prepare(original_cfg)  # No run_arm/test/training.
    if screen.base.m2._feature_hash(prepared['features']) != prepared['feature_sha256']:
        raise RuntimeError('고정 특징 해시 불일치')
    path = partition.CONTROL/'verified_gat_inputs.pt.gz'
    if path.exists():
        raise RuntimeError('기존 입력 스냅샷은 덮어쓰지 않습니다')
    with gzip.open(path.with_suffix('.tmp'), 'wb', compresslevel=1) as output:
        torch.save(prepared, output)
    path.with_suffix('.tmp').replace(path)
    manifest_path = partition.CONTROL/'l4_handoff.json'
    manifest = json.loads(manifest_path.read_text())
    platform.atomic_json(partition.CONTROL/'l4_handoff_before_snapshot.json',manifest)
    manifest.update(wrapper_sha256=platform.sha(partition.__file__),
        input_loader_sha256=platform.sha(__file__),
        frozen_input_sha256=platform.sha(path),
        frozen_input_protocol_sha256=partition.digest(prepared['protocol']),
        frozen_input_note='untrained inputs/CLV/features/graphs only; original L4 reference hash checks passed; no learned parameters')
    platform.atomic_json(manifest_path,manifest)
    print('VERIFIED_UNTRAINED_INPUTS_READY',path.stat().st_size,manifest['frozen_input_sha256'],flush=True)


def install(screen, manifest, root):
    import torch
    import dh_gat_partition as partition
    if platform.sha(__file__) != manifest['input_loader_sha256']:
        raise RuntimeError('고정 입력 로더 변경 감지')
    path = partition.CONTROL/'verified_gat_inputs.pt.gz'

    def prepare(cfg):
        screen.original.validate_config(cfg)
        shared = read_snapshot(path,manifest['frozen_input_sha256'])
        if partition.digest(shared['protocol']) != manifest['frozen_input_protocol_sha256']:
            raise RuntimeError('고정 입력 protocol 불일치')
        if screen.base.m2._feature_hash(shared['features']) != shared['feature_sha256']:
            raise RuntimeError('L4 M2 특징 바이트 변경')
        import hashlib
        weights = shared['observed_weights']
        if hashlib.sha256(weights.tobytes()).hexdigest() != shared['protocol']['m4_weight_audit']['sha256']:
            raise RuntimeError('L4 M4 가중치 바이트 변경')
        if shared['revision'] != platform.JOBS['dh-gat']['pin']:
            raise RuntimeError('고정 입력 소스 불일치')
        # The new runner builds a fresh per-seed model and optimizer below.
        # Only physical output paths change; the pinned remaining9 prepare
        # still checks all original seed49 source/input/feature/M4/audit gates.
        shared.update(cfg=cfg,out_dir=Path(cfg.out_dir))
        shared['data']['adj'] = shared['data']['adj'].to(screen.base.v3.DEVICE)
        if 'observed_adj' in shared:
            shared['observed_adj'] = shared['observed_adj'].to(screen.base.v3.DEVICE)
        print('L4의 검증된 미학습 전처리 입력 재사용; 원본 특징·M4 해시 유지',flush=True)
        return shared

    screen.original.prepare = prepare


if __name__ == '__main__':
    if sys.argv[1:] != ['export']:
        raise SystemExit('L4에서 export만 허용')
    export()
