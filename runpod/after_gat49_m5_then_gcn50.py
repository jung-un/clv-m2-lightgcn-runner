"""On 4mgakh8utyoyqu only: completed GAT49 M5 -> H&M GCN50 eight arms.

No polling Drive, no new Pod, no interruption of the prior training. A failed
prior job blocks the queue. Re-run `run` from the same pinned checkout to resume.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys

import launch as platform
import hm_ngcf_m5 as storage

POD = '4mgakh8utyoyqu'
SHARED = Path('/workspace/clv')
PRIOR = Path('/workspace/clv_hm_gat_m5_seed49')
ROOT = Path('/workspace/clv_hm_gcn_seed50_all8')
JOB = 'hm-gcn'
MODELS = platform.ORDER
PRIOR_ARGS = ['-u', '/workspace/hm_gat_m5_after_ngcf/after_ngcf_then_gat.py',
              'wait', '--ngcf-pid', '7796']


def process_identity(pid):
    try:
        p = Path('/proc') / str(pid)
        args = (p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        ticks = (p/'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None
    if args[1:] != PRIOR_ARGS:
        raise RuntimeError('대기 PID가 확인한 GAT49 M5 예약 프로세스와 다릅니다')
    return ticks


def completion_check(prior=PRIOR):
    status = json.loads((prior/'queue_status.json').read_text())
    if status.get('phase') != 'completed_gat_m5':
        raise RuntimeError('GAT49 M5_A~D 정상 종료 확인 전 GCN 시작 금지')
    output = prior/'results/hm-gat'
    assignment = json.loads((output/'runpod_m5_partition.json').read_text())
    if assignment.get('seeds') != [49] or assignment.get('models') != list(MODELS[4:]):
        raise RuntimeError('GAT 선행 배정 불일치')
    protocols = list(output.glob('*/protocol.json'))
    if len(protocols) != 1:
        raise RuntimeError('선행 GAT 실행 폴더가 유일하지 않습니다')
    run = protocols[0].parent
    protocol = json.loads(protocols[0].read_text())
    if protocol['config']['seeds'] != [49] or protocol['config']['epochs'] != 300:
        raise RuntimeError('선행 시드/epoch 불일치')
    for label in MODELS[4:]:
        mid = f'gat_{label.lower()}_lastweek'
        expected = dict(stage=protocol['code_version'], model_id=mid, seed=49,
                        config_hash=run.name, source_revision=protocol['source_revision'],
                        input_hash=protocol['input_hash'])
        row = json.loads((run/'arms'/f'{mid}_s49.json').read_text())
        marker = json.loads((run/'progress/stages'/f"{protocol['code_version']}_{mid}_s49.completed.json").read_text())
        if (row.get('identity') != expected or row.get('model') != label
                or row.get('seed') != 49 or row.get('epochs') != 300
                or [r.get('epoch') for r in row.get('training_history', [])] != list(range(1,301))
                or len(row.get('metrics', {})) != 142
                or not all(math.isfinite(v) for v in row['metrics'].values())
                or marker.get('identity') != expected or marker.get('status') != 'completed'):
            raise RuntimeError(f'선행 {label} 완료 원본 검증 실패')
    return protocol


def preflight():
    if os.environ.get('RUNPOD_POD_ID') != POD:
        raise RuntimeError('승인된 Pod 전용입니다')
    repo = Path(__file__).resolve().parents[1]
    pin = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'], text=True).strip():
        raise RuntimeError('고정 소스가 변경되었습니다')
    storage.link_shared_inputs(ROOT, SHARED)
    platform.JOBS[JOB] = dict(module='clv_hm_gcn_final8_seed50', dataset='hm', pin=pin, seed=50)
    target = ROOT/'sources'/pin
    if not target.exists():
        target.symlink_to(repo, target_is_directory=True)
    platform.source_checkout(JOB, ROOT)
    inputs = platform.verify_files(JOB, ROOT)
    platform.guard_space(ROOT)
    sys.path.insert(0, str(repo))
    return pin, inputs


def status(phase, **extra):
    platform.atomic_json(ROOT/'queue_status.json', dict(
        phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
        pod=POD, next_job=JOB, seed=50, models=list(MODELS), **extra))
    print(phase, extra, flush=True)


def run():
    pin, inputs = preflight()
    completion_check()
    with platform.exclusive(SHARED/'locks/hm-gcn-seed50-all8.lock'):
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU에 기존 작업이 있어 새 학습을 시작하지 않습니다')
        import torch
        platform.cuda_check(torch)
        screen, cfg = platform.load_runner(JOB, ROOT)
        output = ROOT/'results'/JOB
        env = platform.environment(torch, JOB, inputs)
        platform.bind_environment(output/'runpod_environment.json', env)
        storage.install_storage(screen, output)
        prepared = screen.prepare(cfg)
        if cfg.seeds != (50,) or cfg.epochs != 300 or tuple(screen.ORDER) != MODELS:
            raise RuntimeError('고정 seed50/300epoch/8모형 위반')
        if set(prepared['data']['splits']) != {'test'} or prepared['data'].get('loss_w') is not None:
            raise RuntimeError('최종 test/기본 비가중 손실 조건 위반')
        platform.atomic_json(output/'runpod_assignment.json', dict(
            seed=50, models=list(MODELS), source_commit=pin,
            predecessor='GAT seed49 M5_A~D; scheduling only, no pretrained weights',
            checkpoint_policy='every epoch atomic; latest and previous; compact completed only'))
        exports = ROOT/'exports'
        exports.mkdir(parents=True, exist_ok=True)
        for index, label in enumerate(MODELS, 1):
            print(f'[{index}/8] H&M GCN seed50 {label}', flush=True)
            with platform.evaluation_once(screen, prepared, label, 50):
                screen.run_arm(prepared, label)
            path = screen.export_zip(prepared, str(exports/'hm_gcn_seed50_results.zip'))
            print('RESULT_ZIP:', path, flush=True)
        result = screen.report(prepared)
        if not result['complete']:
            raise RuntimeError('8모형 완료 결과 누락')
        print('GCN_SEED50_ALL8_COMPLETE. Pod 과금은 계속됩니다.', flush=True)


def queue(mode, pid):
    with platform.exclusive(SHARED/'locks/gat49-m5-to-gcn50-all8.queue.lock'):
        try:
            pin, _ = preflight()
            ticks = process_identity(pid)
            if ticks is None:
                completion_check()
            if mode == 'check':
                print('GCN_QUEUE_PREFLIGHT_OK', dict(prior_pid=pid, waiting=ticks is not None,
                      source_commit=pin, seed=50, models=MODELS), flush=True)
                return
            status('waiting_for_gat49_m5_completion', prior_pid=pid, process_start_ticks=ticks, source_commit=pin)
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
            log = ROOT/'gcn_seed50_all8_run.log'
            with log.open('ab') as out:
                child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],
                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                status('running_gcn_seed50_all8', child_pid=child.pid, log=str(log))
                code = child.wait()
            if code:
                raise RuntimeError(f'GCN 실행 실패({code}); 자동 재시도 없음, checkpoint 보존')
            status('completed_gcn_seed50_all8', log=str(log))
        except Exception as exc:
            status('failed_no_automatic_retry', error=str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check','wait','run'))
    parser.add_argument('--prior-pid', type=int, default=9345)
    args = parser.parse_args()
    if args.mode == 'run':
        run()
    else:
        queue(args.mode, args.prior_pid)

