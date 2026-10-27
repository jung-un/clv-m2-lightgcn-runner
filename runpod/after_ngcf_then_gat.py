"""One-shot NGCF M1~M4 completion -> GAT M5 queue on the existing NGCF Pod."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import subprocess

import launch as platform
import hm_gat_m5 as gat

POD = '4mgakh8utyoyqu'
NGCF_PIN = 'db39115fadad5aedb020eafe4a492a2862a876db'
RUN_ID = 'e4c90cf5ad8d'
MODELS = ('M1','M2','M3','M4')
ROOT = Path('/workspace/clv_hm_gat_m5_seed49')
SHARED = Path('/workspace/clv')


def process_identity(pid, proc=Path('/proc')):
    path = proc / str(pid)
    try:
        args = (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        ticks = (path / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None
    if not (len(args) > 4 and args[1:4] == ['-u','/workspace/hm_ngcf_fixed/launch.py','run']
            and args[4:] == ['--job','hm-ngcf','--seeds','49','--models',*MODELS,'--confirmed-unstarted']):
        raise RuntimeError('기다릴 PID가 기존 NGCF M1~M4 프로세스와 다릅니다')
    return ticks


def completion_check(shared=SHARED):
    run = shared / 'results/hm-ngcf' / RUN_ID
    protocol = json.loads((run / 'protocol.json').read_text())
    if protocol['source_revision'] != NGCF_PIN or protocol['config']['epochs'] != 300:
        raise RuntimeError('NGCF 실행 신원/고정 epoch 불일치')
    records = sorted((shared / 'results/hm-ngcf').glob('runpod_invocation_*.json'))
    invocation = json.loads(records[-1].read_text())
    if invocation.get('seeds') != [49] or invocation.get('models') != list(MODELS):
        raise RuntimeError('NGCF의 담당 범위가 변경됐습니다')
    for label in MODELS:
        model_id = f'ngcf_{label.lower()}_lastweek'
        row = json.loads((run / 'arms' / f'{model_id}_s49.json').read_text())
        expected = dict(stage=protocol['code_version'], model_id=model_id, seed=49,
                        config_hash=RUN_ID, source_revision=NGCF_PIN, input_hash=protocol['input_hash'])
        if (row.get('identity') != expected or row.get('model') != label or row.get('seed') != 49
                or row.get('epochs') != 300
                or [x.get('epoch') for x in row.get('training_history', [])] != list(range(1,301))
                or len(row.get('metrics', {})) != 142
                or not all(math.isfinite(v) for v in row['metrics'].values())):
            raise RuntimeError(f'{label}의 정상 완료 결과가 확인되지 않습니다')
        stage = run / 'progress/stages' / f"{protocol['code_version']}_{model_id}_s49.completed.json"
        if not stage.is_file():
            raise RuntimeError(f'{label} 저장 완료 표식 없음')
        marker = json.loads(stage.read_text())
        if marker.get('identity') != expected or marker.get('status') != 'completed':
            raise RuntimeError(f'{label} 저장 완료 표식 신원 불일치')
    log = shared.parent / 'hm_resume.log'
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size-8192))
        tail = f.read().decode(errors='replace')
    if '요청한 작업 완료.' not in tail:
        raise RuntimeError('NGCF 정상 종료 로그가 없어 후속 학습을 시작하지 않습니다')


def status(root, phase, **extra):
    platform.atomic_json(root / 'queue_status.json', {
        'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'pod': POD, 'next_job': 'hm-gat', 'seed': 49, 'models': list(gat.MODELS), **extra})
    print(phase, json.dumps(extra, ensure_ascii=False), flush=True)


def wait_for_process(pid, identity):
    # pidfd tracks this exact process, including through PID reuse.
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        current = process_identity(pid)
        if current is not None and current != identity:
            raise RuntimeError('NGCF PID가 재사용되었습니다')
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        while not poll.poll(30_000):
            pass
    finally:
        os.close(fd)


def queue(mode, pid, root=ROOT, shared=SHARED):
    if os.environ.get('RUNPOD_POD_ID') != POD:
        raise RuntimeError('기존 NGCF Pod에서만 이 후속 실행을 허용합니다')
    root.mkdir(parents=True, exist_ok=True)
    with platform.exclusive(shared / 'locks/hm-ngcf-to-gat-m5.queue.lock'):
        try:
            gat.baseline(shared)  # No GPU, preprocessing or training while waiting.
            identity = process_identity(pid)
            if identity is None:
                completion_check(shared)
            if mode == 'check':
                print('QUEUE_PREFLIGHT_OK', {'ngcf_pid': pid, 'ngcf_active': identity is not None,
                                            'next_job': 'hm-gat', 'models': gat.MODELS}, flush=True)
                return
            status(root, 'waiting_for_ngcf_completion', ngcf_pid=pid, process_start_ticks=identity)
            if identity is not None:
                wait_for_process(pid, identity)
            with platform.exclusive(shared / 'locks/hm-ngcf.lock'):
                completion_check(shared)
                gat.baseline(shared)
                active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
                if active:
                    raise RuntimeError('이 GPU에 다른 계산 프로세스가 남아 있습니다; 자동 시작 차단')
            if os.environ.get('RUNPOD_POD_ID') != POD:
                raise RuntimeError('대상 Pod 변경')
            log = root / 'gat_m5_run.log'
            with log.open('ab') as out:
                child = subprocess.Popen([str(shared/'venv/bin/python'),'-u',
                    str(Path(__file__).with_name('hm_gat_m5.py')),'run','--confirmed-unstarted'],
                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                status(root, 'running_gat_m5', child_pid=child.pid, log=str(log))
                code = child.wait()
            if code:
                raise RuntimeError(f'GAT M5 실행 종료 코드 {code}; 로그를 확인하세요')
            status(root, 'completed_gat_m5', log=str(log))
        except Exception as exc:
            status(root, 'failed_no_automatic_retry', error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('check','wait'))
    p.add_argument('--ngcf-pid', required=True, type=int)
    args = p.parse_args()
    queue(args.mode, args.ngcf_pid)


if __name__ == '__main__':
    main()
