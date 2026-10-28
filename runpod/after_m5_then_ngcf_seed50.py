"""One-shot on the existing M5 Pod: NGCF49 M5 completion -> NGCF50 M1~M4."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import subprocess

import launch as platform
import hm_ngcf_m5 as storage

POD = 'rkdcwyt2m28a1b'
SHARED = Path('/workspace/clv')
PRIOR_ROOT = Path('/workspace/clv_hm_ngcf_m5_seed49')
ROOT = Path('/workspace/clv_hm_ngcf_seed50_core')
PRIOR_RUN = '5da20063d68b'
PIN = 'f3ebca93654c192b5dd95ce6f6cd55b588d78c9c'
MODELS = ('M1', 'M2', 'M3', 'M4')
PRIOR_MODELS = ('M5_A', 'M5_B', 'M5_C', 'M5_D')


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid)
        args = (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        ticks = (path / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None
    if args[1:] != ['-u', '/workspace/hm_ngcf_m5_start/hm_ngcf_m5.py',
                     'run', '--confirmed-unstarted']:
        raise RuntimeError('대기 PID가 기존 NGCF seed49 M5 작업과 다릅니다')
    return ticks


def completion_check(prior=PRIOR_ROOT):
    output = prior / 'results/hm-ngcf'
    assignment = json.loads((output / 'runpod_m5_partition.json').read_text())
    if assignment['seeds'] != [49] or assignment['models'] != list(PRIOR_MODELS):
        raise RuntimeError('기존 M5 담당 범위 변경')
    run = output / PRIOR_RUN
    protocol = json.loads((run / 'protocol.json').read_text())
    if protocol['source_revision'] != storage.PIN or protocol['config']['epochs'] != 300:
        raise RuntimeError('기존 NGCF seed49 실행 신원 불일치')
    for label in PRIOR_MODELS:
        mid = f'ngcf_{label.lower()}_lastweek'
        expected = dict(stage=protocol['code_version'], model_id=mid, seed=49,
                        config_hash=PRIOR_RUN, source_revision=storage.PIN,
                        input_hash=protocol['input_hash'])
        row = json.loads((run / 'arms' / f'{mid}_s49.json').read_text())
        marker = json.loads((run / 'progress/stages' /
            f"{protocol['code_version']}_{mid}_s49.completed.json").read_text())
        if (row.get('identity') != expected or row.get('model') != label
                or row.get('seed') != 49 or row.get('epochs') != 300
                or [x.get('epoch') for x in row.get('training_history', [])] != list(range(1,301))
                or len(row.get('metrics', {})) != 142
                or not all(math.isfinite(v) for v in row['metrics'].values())
                or marker.get('identity') != expected or marker.get('status') != 'completed'):
            raise RuntimeError(f'{label} 정상 완료·저장 검증 실패')
    logs = list((prior / 'logs').glob('*.log'))
    if not logs or 'M5_A~D 요청 완료.' not in max(logs, key=lambda p:p.stat().st_mtime).read_text()[-8192:]:
        raise RuntimeError('기존 M5 정상 종료 로그 없음')
    return protocol


def check_science(current, baseline):
    # Only the driver gained seed plumbing; all model/training module hashes stay fixed.
    keys = tuple(k for k in storage.SCIENCE_KEYS if k not in ('source_revision', 'code_sha256'))
    for k in keys:
        if json.loads(json.dumps(current[k])) != baseline[k]:
            raise RuntimeError('seed50 고정 설계 불일치: ' + k)
    hashes = {k:v for k,v in current['code_sha256'].items() if k != 'clv_hm_ngcf_final8_seed49.py'}
    expected = {k:v for k,v in baseline['code_sha256'].items() if k != 'clv_hm_ngcf_final8_seed49.py'}
    if hashes != expected:
        raise RuntimeError('백본/M2/저장/학습 코드 변경')
    a = {k:v for k,v in current['config'].items() if k not in ('out_dir','m2_reference','m4_reference','seeds')}
    b = {k:v for k,v in baseline['config'].items() if k not in ('out_dir','m2_reference','m4_reference','seeds')}
    if a != b or current['config']['seeds'] != (50,) or baseline['config']['seeds'] != [49]:
        raise RuntimeError('seed 외 학습조건 변경 또는 시드 신원 오류')


def preflight(root=ROOT, shared=SHARED):
    if os.environ.get('RUNPOD_POD_ID') != POD:
        raise RuntimeError('승인된 NGCF M5 Pod에서만 실행합니다')
    storage.link_shared_inputs(root, shared)
    platform.JOBS['hm-ngcf'] = dict(platform.JOBS['hm-ngcf'], pin=PIN, seed=50)
    platform.verify_files('hm-ngcf', root)
    platform.source_checkout('hm-ngcf', root)
    platform.guard_space(root)


def run_seed50(root=ROOT, shared=SHARED):
    preflight(root, shared)
    with platform.exclusive(shared / 'locks/hm-ngcf-seed50-core.lock'):
        baseline = completion_check()
        import torch
        platform.cuda_check(torch)
        screen, default = platform.load_runner('hm-ngcf', root)
        cfg = replace(default, seeds=(50,))
        screen.validate_config(cfg)
        inputs = platform.verify_files('hm-ngcf', root)
        env = platform.environment(torch, 'hm-ngcf', inputs)
        prior_env = json.loads((PRIOR_ROOT/'results/hm-ngcf/runpod_environment.json').read_text())
        for key in ('input_sha256','python','packages','cuda','gpu','capability','tf32_matmul','matmul_precision'):
            if env[key] != prior_env[key]:
                raise RuntimeError('기존 NGCF 환경과 다릅니다: ' + key)
        output = root / 'results/hm-ngcf'
        platform.bind_environment(output/'runpod_environment.json', env)
        storage.install_storage(screen, output)
        prepared = screen.prepare(cfg)
        check_science(prepared['protocol'], baseline)
        if set(prepared['data']['splits']) != {'test'} or prepared['data'].get('loss_w') is not None:
            raise RuntimeError('고정 test/기본 BPR 조건 불일치')
        platform.atomic_json(output/'runpod_seed50_partition.json', {
            'seed':50, 'models':list(MODELS), 'source_commit':PIN,
            'seed49_role':'input/design audit only; no scores or learned weights reused',
            'other_models':'M5_A~D unassigned here', 'root':str(root)})
        for label in MODELS:
            print(f'[실행] H&M NGCF seed50 {label}', flush=True)
            with platform.evaluation_once(screen, prepared, label, 50):
                screen.run_arm(prepared, label)
            platform.export(screen, prepared, 'hm-ngcf', root)
        if any(screen.completed(prepared, label) is None for label in MODELS):
            raise RuntimeError('seed50 M1~M4 완료 결과 누락')
        print('SEED50_CORE_COMPLETE: M1~M4 완료; M5 학습 없음; Pod 과금은 계속됩니다.', flush=True)


def status(phase, **extra):
    platform.atomic_json(ROOT/'queue_status.json', dict(
        phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
        pod=POD, next_job='hm-ngcf', seed=50, models=list(MODELS), **extra))
    print(phase, extra, flush=True)


def queue(mode, pid):
    with platform.exclusive(SHARED/'locks/ngcf49-m5-to-ngcf50-core.queue.lock'):
        try:
            preflight()
            ticks = process_identity(pid)
            if ticks is None:
                completion_check()
            if mode == 'check':
                print('QUEUE_PREFLIGHT_OK', {'pid':pid, 'waiting':ticks is not None,
                                             'seed':50, 'models':MODELS}, flush=True)
                return
            status('waiting_for_ngcf49_m5_completion', prior_pid=pid, process_start_ticks=ticks)
            if ticks is not None:
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    fd = None
                if fd is not None:
                    try:
                        if process_identity(pid) not in (None, ticks):
                            raise RuntimeError('대기 PID 재사용')
                        poll = select.poll()
                        poll.register(fd, select.POLLIN)
                        while not poll.poll(30_000):
                            pass
                    finally:
                        os.close(fd)
            completion_check()
            active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
            if active:
                raise RuntimeError('GPU에 다른 계산 프로세스가 있어 후속 실행 차단')
            log = ROOT/'seed50_core_run.log'
            with log.open('ab') as out:
                child = subprocess.Popen([str(SHARED/'venv/bin/python'),'-u',str(Path(__file__).resolve()),'run'],
                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                status('running_ngcf_seed50_core', child_pid=child.pid, log=str(log))
                code = child.wait()
            if code:
                raise RuntimeError(f'seed50 실행 종료 코드 {code}; 자동 재시도 없음')
            status('completed_ngcf_seed50_core', log=str(log))
        except Exception as exc:
            status('failed_no_automatic_retry', error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('check','wait','run'))
    p.add_argument('--prior-pid', type=int, default=9398)
    args = p.parse_args()
    if args.mode == 'run':
        run_seed50()
    else:
        queue(args.mode, args.prior_pid)


if __name__ == '__main__':
    main()
