"""Separate H&M NGCF seed49 M5 partition; never modifies the M1~M4 job."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import launch as platform

SHARED = Path('/workspace/clv')
ROOT = Path('/workspace/clv_hm_ngcf_m5_seed49')
MODELS = ('M5_A', 'M5_B', 'M5_C', 'M5_D')
PIN = 'db39115fadad5aedb020eafe4a492a2862a876db'
SCIENCE_KEYS = ('source_revision', 'code_sha256', 'input_hash', 'data_stats', 'split',
                'feature_sha256', 'm4_weight_audit', 'fixed_intervention_settings',
                'models', 'backbone', 'propagation', 'adjacency', 'regularization',
                'initialization', 'fixed_epoch', 'validation', 'holdout', 'early_stopping')


def check_existing(shared=SHARED):
    """Read-only: require a recorded disjoint assignment and no prior M5 work."""
    output = shared / 'results/hm-ngcf'
    invocations = sorted(output.glob('runpod_invocation_*.json'))
    if not invocations:
        raise RuntimeError('기존 M1~M4 실행 배정 기록을 찾을 수 없습니다. 같은 영구볼륨을 연결하세요.')
    assignment = json.loads(invocations[-1].read_text())
    if (assignment.get('seeds') != [49]
            or assignment.get('models') != ['M1', 'M2', 'M3', 'M4']):
        raise RuntimeError('기존 실행이 seed49 M1~M4 전용으로 확인되지 않습니다. 중복 방지를 위해 중단합니다.')
    for path in output.rglob('*'):
        if re.search(r'(?:ngcf_)?m5_[abcd](?:_|\.)', path.name.lower()):
            raise RuntimeError(f'기존 출력에 혼합모형 기록이 있습니다. 재실행하지 않습니다: {path}')
    protocols = [json.loads(p.read_text()) for p in output.glob('*/protocol.json')]
    protocols = [p for p in protocols if p.get('source_revision') == PIN]
    if not protocols:
        raise RuntimeError('기존 NGCF와 같은 고정 커밋의 protocol을 찾을 수 없습니다.')
    return protocols


def link_shared_inputs(root=ROOT, shared=SHARED):
    if root.resolve() == shared.resolve() or shared.resolve() in root.resolve().parents:
        raise RuntimeError('새 출력 root는 기존 실행 폴더와 분리해야 합니다.')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('data', 'sources'):
        target, source = root / name, shared / name
        if not source.is_dir():
            raise RuntimeError(f'기존 검증 입력/소스가 없습니다: {source}')
        if target.is_symlink() or target.exists():
            if target.resolve() != source.resolve():
                raise RuntimeError(f'다른 입력/소스 경로를 덮어쓰지 않습니다: {target}')
        else:
            target.symlink_to(source, target_is_directory=True)
    if not (root / 'sources' / PIN).is_dir():
        raise RuntimeError('기존 고정 소스가 없습니다. 기존 서버의 코드/환경을 변경하지 않습니다.')


def scientific_identity(protocol):
    values = {key: protocol[key] for key in SCIENCE_KEYS}
    values['config'] = {k: v for k, v in protocol['config'].items()
                        if k not in ('out_dir', 'm2_reference', 'm4_reference')}
    # Live dataclass tuples become lists in stored JSON; compare persisted values.
    return json.loads(json.dumps(values, allow_nan=False))


def check_science(protocol, baselines):
    actual = scientific_identity(protocol)
    if any(scientific_identity(p) != actual for p in baselines):
        raise RuntimeError('기존 M1~M4와 학습/입력/분할/모형 신원이 다릅니다. 학습하지 않습니다.')


def check_environment(current, shared):
    previous = json.loads((shared / 'results/hm-ngcf/runpod_environment.json').read_text())
    keys = ('source_commit', 'input_sha256', 'python', 'packages', 'cuda', 'gpu',
            'capability', 'tf32_matmul', 'matmul_precision')
    if any(previous[k] != current[k] for k in keys):
        raise RuntimeError('기존 NGCF와 GPU/패키지/정밀도 환경이 다릅니다. 조건을 확인한 뒤 진행하세요.')


def preserve_previous(latest):
    """Original writer replaces latest atomically, so hard links retain old bytes."""
    previous = latest.with_suffix('.previous.pt')
    temporary = previous.with_suffix(previous.suffix + '.tmp')
    if temporary.exists():
        raise RuntimeError(f'미완료 보존 파일이 있습니다. 복구 확인 전 덮어쓰지 않습니다: {temporary}')
    try:
        os.link(latest, temporary)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS):
            raise
        # Unsupported network filesystems retain the old verified-copy behavior.
        import hashlib
        digest = hashlib.sha256()
        with latest.open('rb') as source, temporary.open('xb') as dest:
            for block in iter(lambda: source.read(8 * 1024**2), b''):
                digest.update(block)
                dest.write(block)
            dest.flush()
            os.fsync(dest.fileno())
        if platform.sha(temporary) != digest.hexdigest():
            raise RuntimeError('직전 checkpoint 복사 검증 실패; 기존 파일 보존')
        method = 'verified_copy'
    else:
        if not os.path.samefile(latest, temporary):
            raise RuntimeError('직전 checkpoint 동일 파일 확인 실패')
        method = 'hardlink'
    os.replace(temporary, previous)
    return method


def install_storage(screen, output):
    original = screen.SafeProgressStore

    class FastStore(original):
        def save_epoch(self, model, optimizer, rng, **state):
            estimate = sum(p.numel() * p.element_size() for p in model.parameters()) * 3
            latest_bytes = self.latest_checkpoint.stat().st_size if self.latest_checkpoint.exists() else 0
            platform.guard_space(output, estimate + latest_bytes + 512 * 1024**2)
            started = time.perf_counter()
            method = preserve_previous(self.latest_checkpoint) if self.latest_checkpoint.exists() else 'first_epoch'
            elapsed = time.perf_counter() - started
            result = super().save_epoch(model, optimizer, rng, **state)
            self.last_save_timing.update(previous_checkpoint_method=method,
                                         previous_checkpoint_seconds=elapsed)
            return result

    screen.SafeProgressStore = FastStore
    screen.STORAGE_GUARD = lambda n: platform.guard_space(output, n)


def execute(mode, root=ROOT, shared=SHARED):
    if platform.JOBS['hm-ngcf']['pin'] != PIN:
        raise RuntimeError('동봉 launch.py의 NGCF 고정 커밋 불일치')
    # A common lock also catches this package running twice on the shared volume.
    with platform.exclusive(shared / 'locks/hm-ngcf-seed49-m5-partition.lock'):
        baselines = check_existing(shared)
        link_shared_inputs(root, shared)
        import torch
        inputs = platform.verify_files('hm-ngcf', root)
        platform.cuda_check(torch)
        screen, cfg = platform.load_runner('hm-ngcf', root)
        output = root / 'results/hm-ngcf'
        platform.guard_space(output)
        env = platform.environment(torch, 'hm-ngcf', inputs)
        check_environment(env, shared)
        env.update(partition='seed49-M5_A-to-D', partition_launcher_sha256=platform.sha(__file__),
                   checkpoint_policy='atomic-publish-each-epoch; hardlink-previous-or-verified-copy')
        if mode == 'check':
            print(json.dumps(env, ensure_ascii=False, indent=2), flush=True)
            print('배정·입력·고정 소스·CUDA 점검 통과. 학습/전체 전처리는 아직 시작하지 않았습니다.')
            return
        platform.bind_environment(output / 'runpod_environment.json', env)
        install_storage(screen, output)
        prepared = screen.prepare(cfg)  # One preparation shared by all four arms.
        check_science(prepared['protocol'], baselines)
        if (set(prepared['data']['splits']) != {'test'}
                or prepared['data'].get('loss_w') is not None
                or prepared['data']['data_stats']['min_item_inter'] != 1):
            raise RuntimeError('고정 test/k1/기본BPR 조건 불일치')
        if mode == 'report':
            platform.export(screen, prepared, 'hm-ngcf', root)
            return
        platform.atomic_json(output / 'runpod_m5_partition.json', {
            'seeds': [49], 'models': list(MODELS), 'source_commit': PIN,
            'original_output_read_only': str(shared / 'results/hm-ngcf'),
            'separate_root': str(root), 'started_utc': datetime.now(timezone.utc).isoformat()})
        for label in MODELS:
            check_existing(shared)
            print(f'[실행] hm-ngcf seed49 {label} — 새 GPU 전용', flush=True)
            with platform.evaluation_once(screen, prepared, label, 49):
                screen.run_arm(prepared, label)
            platform.export(screen, prepared, 'hm-ngcf', root)
        print('M5_A~D 요청 완료. 기존 M1~M4 결과와의 취합은 별도 검증 후 수행합니다. Pod 과금은 계속됩니다.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'run', 'report'))
    parser.add_argument('--confirmed-unstarted', action='store_true')
    args = parser.parse_args()
    if args.mode == 'run' and not args.confirmed_unstarted:
        parser.error('다른 볼륨·Colab에 같은 M5 seed49 실행이 없는지 확인하고 --confirmed-unstarted를 지정하세요.')
    execute(args.mode)


if __name__ == '__main__':
    main()
