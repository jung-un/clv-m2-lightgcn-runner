"""Fixed L4/Ada seed ownership; original pinned GAT learning stays unchanged.

Use beside the original 2026-10-26 launch.py. L4 is handed off first;
copy its handoff JSON to Ada, then run `ada-check` and `ada-run`.
Different volumes cannot provide a distributed lock: the verified, reduced
L4 invocation is the ownership boundary. Never restart its old full-seed command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import launch as platform

L4_SEEDS = (43, 44, 45)
ADA_SEEDS = (46, 47, 48, 50, 51)
OLD_SEEDS = (42, 43, 44, 45, 46, 47, 48, 50, 51)
L4_ROOT = Path('/workspace/clv')
ADA_ROOT = Path('/workspace/clv_dh_gat_ada')
RUN_ID = 'c3832d21f165'
OLD_LAUNCHER_SHA = 'f93f3f25f8222cd4c563e76a7766c54ce17060da835dec7ddb30c18215bba9fc'
CONTROL = Path('/workspace/dh_gat_partition_20261027')
IDENTITY_KEYS = (
    'code_version', 'source_revision', 'code_sha256', 'input_hash', 'data_stats',
    'split', 'feature_sha256', 'dataset_initialization',
    'fixed_intervention_settings', 'models', 'backbone', 'propagation',
    'adjacency', 'regularization', 'implementation_scope', 'm2_adaptation',
    'initialization', 'fixed_epoch', 'test_evaluations_per_completed_fit',
    'execution_optimization', 'validation', 'holdout', 'early_stopping',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def scientific_identity(protocol):
    result = {key: protocol[key] for key in IDENTITY_KEYS}
    result['config'] = {k: v for k, v in protocol['config'].items()
                        if k not in ('out_dir', 'm2_reference', 'm4_reference', 'seed49_reference')}
    return json.loads(json.dumps(result, allow_nan=False))


def validate_manifest(manifest):
    if manifest.get('l4_seeds') != list(L4_SEEDS) or manifest.get('ada_seeds') != list(ADA_SEEDS):
        raise RuntimeError('고정 시드 배정 불일치')
    if manifest.get('state') != 'l4_restricted_and_resumed':
        raise RuntimeError('L4 시드 제한 및 재개가 확인되지 않았습니다')
    if manifest.get('source') != platform.JOBS['dh-gat']['pin']:
        raise RuntimeError('고정 학습 소스 불일치')
    if manifest.get('launcher_sha256') != OLD_LAUNCHER_SHA:
        raise RuntimeError('기존 실행도구 불일치')
    if manifest.get('wrapper_sha256') != platform.sha(__file__):
        raise RuntimeError('분담 실행도구 변경 감지')
    if manifest.get('resumed_epoch', 0) <= manifest.get('checkpoint_epoch', 0):
        raise RuntimeError('실제 재개 epoch 증거 없음')


def process_args(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')


def expected_args(seeds):
    return [str(L4_ROOT/'venv/bin/python'), '-u', '/workspace/runpod_gat/launch.py',
            'run', '--job', 'dh-gat', '--seeds', *map(str, seeds), '--confirmed-unstarted']


def validate_checkpoint(payload, progress, checksum):
    required = {'identity', 'model_state', 'optimizer_state', 'numpy_rng_state',
                'torch_rng_state', 'cuda_rng_state', 'epoch', 'history'}
    if not required.issubset(payload) or not payload['cuda_rng_state']:
        raise RuntimeError('모델/optimizer/난수 저장 상태 누락')
    identity = payload['identity']
    epoch = int(payload['epoch'])
    if identity['seed'] not in L4_SEEDS or identity['config_hash'] != RUN_ID:
        raise RuntimeError('현재 L4 checkpoint 신원 불일치')
    if not 1 <= epoch < 295 or int(progress['epoch']) != epoch:
        raise RuntimeError('평가 인접 지점 또는 저장 진행정보 불일치: 기존 실행 유지')
    if [int(row['epoch']) for row in payload['history']] != list(range(1, epoch + 1)):
        raise RuntimeError('checkpoint의 epoch 이력 불연속')
    if progress.get('checkpoint_sha256') != checksum:
        raise RuntimeError('checkpoint 해시 불일치')
    return epoch, identity


def rescope_l4(pid):
    """Pause only the verified worker, validate durable state, then resume subset."""
    import torch
    if os.environ.get('RUNPOD_POD_ID') != 'w05yjhfwulnyr4':
        raise RuntimeError('대상 L4 Pod가 아닙니다')
    if process_args(pid) != expected_args(OLD_SEEDS):
        raise RuntimeError('기존 전체 시드 프로세스가 아닙니다. 신호를 보내지 않습니다')
    if platform.sha('/workspace/runpod_gat/launch.py') != OLD_LAUNCHER_SHA:
        raise RuntimeError('기존 launcher 변경 감지')
    run = L4_ROOT/'results/dh-gat'/RUN_ID
    progress_path = run/'progress/progress.json'
    before = json.loads(progress_path.read_text())
    if before['seed'] not in L4_SEEDS or before['epoch'] >= 295:
        raise RuntimeError('현재 모형이 학습 중간 지점이 아닙니다. 기존 실행 유지')
    started = {json.loads(p.read_text()).get('seed') for p in (run/'progress/stages').glob('*.json')}
    # Stage files may nest identity depending on the writer; inspect both forms.
    started |= {json.loads(p.read_text()).get('identity', {}).get('seed')
                for p in (run/'progress/stages').glob('*.json')}
    if started.intersection(ADA_SEEDS):
        raise RuntimeError('Ada 배정 시드의 시작 기록이 존재합니다')
    CONTROL.mkdir(exist_ok=True)
    manifest_path = CONTROL/'l4_handoff.json'
    if manifest_path.exists():
        raise RuntimeError('기존 인계 기록 존재: 재실행하지 않습니다')
    stopped = False
    terminated = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            if Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].startswith('T'):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('프로세스 일시정지 확인 실패')
        if process_args(pid) != expected_args(OLD_SEEDS):
            raise RuntimeError('프로세스 신원 변경')
        progress = json.loads(progress_path.read_text())
        model = progress['model_id']
        checkpoint = run/'progress/resume'/f'clv-dh-gat-remaining9-v1_{model}_s{progress["seed"]}_latest.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        epoch, identity = validate_checkpoint(payload, progress, platform.sha(checkpoint))
        label = next(k for k in platform.ORDER if model == f'gat_{k.lower()}_lastweek')
        if (run/'evaluation_started'/f'{label}_s{progress["seed"]}.json').exists():
            raise RuntimeError('최종평가 시작 기록 존재: 기존 실행 유지')
        protocol = json.loads((run/'protocol.json').read_text())
        manifest = dict(state='checkpoint_verified', l4_seeds=list(L4_SEEDS),
            ada_seeds=list(ADA_SEEDS), completed_reused_seeds=[42,49],
            source=platform.JOBS['dh-gat']['pin'], launcher_sha256=OLD_LAUNCHER_SHA,
            wrapper_sha256=platform.sha(__file__), scientific_sha256=digest(scientific_identity(protocol)),
            environment=json.loads((run.parent/'runpod_environment.json').read_text()),
            checkpoint_sha256=platform.sha(checkpoint), checkpoint_epoch=epoch,
            model_id=model, seed=progress['seed'], old_pid=pid)
        platform.atomic_json(CONTROL/'checkpoint_handoff.json', manifest)
        del payload
        # SIGINT may be ignored by nohup. Queue TERM while stopped, then deliver it.
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        stopped = False
        terminated = True
        for _ in range(600):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split(') ',1)[1].startswith('Z'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('이전 프로세스 종료 미확인: 새 실행 시작 금지')
        # Network-volume lock release can lag the verified process exit.
        for attempt in range(300):
            try:
                with platform.exclusive(L4_ROOT/'locks/dh-gat.lock'):
                    pass
                break
            except RuntimeError:
                if attempt == 299:
                    raise
                time.sleep(.2)
        logfile = CONTROL/'l4_restricted.log'
        with logfile.open('ab', buffering=0) as log:
            child = subprocess.Popen(expected_args(L4_SEEDS), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        manifest['l4_pid'] = child.pid
        manifest['l4_args'] = expected_args(L4_SEEDS)
        platform.atomic_json(CONTROL/'l4_restart_pending.json', manifest)
        print(f'L4_RESTARTED pid={child.pid} checkpoint_epoch={epoch} seeds={L4_SEEDS}', flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f'L4 제한 실행 종료. 로그 확인: {logfile}')
            current = json.loads(progress_path.read_text())
            if (current.get('model_id'),current.get('seed')) == (model,progress['seed']) and current.get('epoch',0) > epoch:
                if process_args(child.pid) != expected_args(L4_SEEDS):
                    raise RuntimeError('제한 실행 인자 불일치')
                manifest.update(state='l4_restricted_and_resumed', resumed_epoch=current['epoch'])
                platform.atomic_json(manifest_path, manifest)
                print('L4_HANDOFF_READY', json.dumps(manifest, ensure_ascii=False), flush=True)
                return
            time.sleep(3)
        raise RuntimeError('10분 내 새 epoch 미확인. 제한 실행은 유지; Ada 시작 금지')
    finally:
        if stopped and not terminated:
            os.kill(pid, signal.SIGCONT)


def ada(mode, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if platform.sha(Path(platform.__file__)) != OLD_LAUNCHER_SHA:
        raise RuntimeError('원래 GAT 실행도구가 아닙니다')
    import torch
    if torch.cuda.get_device_name(0) != 'NVIDIA RTX 6000 Ada Generation':
        raise RuntimeError('승인된 RTX 6000 Ada가 아닙니다')
    inputs = platform.verify_files('dh-gat', ADA_ROOT)
    env = platform.environment(torch, 'dh-gat', inputs)
    for key in ('source_commit','input_sha256','launcher_sha256','python','packages',
                'cuda','tf32_matmul','matmul_precision'):
        if env[key] != manifest['environment'][key]:
            raise RuntimeError(f'L4와 실행 조건 불일치: {key}')
    old_load = platform.load_runner

    def load(job, root):
        screen, cfg = old_load(job, root)
        original_arm = screen.run_arm

        def guarded_arm(prepared, label, seed):
            validate_manifest(json.loads(manifest_path.read_text()))
            if seed not in ADA_SEEDS or label not in platform.ORDER:
                raise RuntimeError('Ada 배정 밖의 작업 차단')
            if digest(scientific_identity(prepared['protocol'])) != manifest['scientific_sha256']:
                raise RuntimeError('L4와 과학적 실행 설정 불일치')
            return original_arm(prepared, label, seed)

        screen.run_arm = guarded_arm
        return screen, cfg

    platform.load_runner = load
    with platform.exclusive(L4_ROOT/'locks/dh-gat-ada-seeds46-48-50-51.lock'):
        output = ADA_ROOT/'results/dh-gat'
        output.mkdir(parents=True, exist_ok=True)
        platform.bind_environment(output/'runpod_partition.json', manifest)
        platform.execute('dh-gat', ADA_SEEDS, mode, root=ADA_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('l4-rescope','ada-check','ada-run'))
    parser.add_argument('--pid', type=int)
    parser.add_argument('--manifest', type=Path, default=CONTROL/'l4_handoff.json')
    args = parser.parse_args()
    if args.mode == 'l4-rescope':
        if not args.pid:
            parser.error('--pid 필요')
        rescope_l4(args.pid)
    else:
        ada('check' if args.mode == 'ada-check' else 'run', args.manifest)


if __name__ == '__main__':
    main()
