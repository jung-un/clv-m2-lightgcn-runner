"""Dunnhumby GAT Ada's five seeds complete -> H&M NGCF50 M5_A~D only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import subprocess

import after_m5_then_ngcf_seed50 as core

platform = core.platform
POD = 'ppfvmn806t4wyf'
ROOT = Path('/workspace/clv_hm_ngcf_seed50_m5')
PRIOR = Path('/workspace/clv_dh_gat_ada/results/dh-gat')
RUN_ID = '6be6705ccaf9'
GAT_PIN = '5a475a2ed25ef0b40822de11c537fa7fc76e475a'
SEEDS = (46,47,48,50,51)
MODELS = ('M5_A','M5_B','M5_C','M5_D')


def process_identity(pid):
    try:
        p = Path('/proc')/str(pid)
        args = (p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        ticks = (p/'stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:
        return None
    if args[1:] != ['-u','/workspace/dh_gat_partition_20261027/dh_gat_partition.py','ada-run']:
        raise RuntimeError('대기할 PID가 기존 Dunnhumby GAT Ada 실행이 아닙니다')
    return ticks


def completion_check(prior=PRIOR, log=Path('/workspace/dh_gat_partition_20261027/ada_run.log')):
    assignment = json.loads((prior/'runpod_partition.json').read_text())
    if assignment.get('ada_seeds') != list(SEEDS) or assignment.get('l4_seeds') != [43,44,45]:
        raise RuntimeError('기존 GAT 시드 분담 변경')
    run = prior/RUN_ID
    protocol = json.loads((run/'protocol.json').read_text())
    if (protocol['source_revision'] != GAT_PIN or protocol['config']['epochs'] != 300
            or protocol['validation'] or protocol['holdout'] or protocol['early_stopping']):
        raise RuntimeError('GAT 고정 실행 설정 불일치')
    for seed in SEEDS:
        for label in platform.ORDER:
            mid = f'gat_{label.lower()}_lastweek'
            ident = dict(stage=protocol['code_version'],model_id=mid,seed=seed,
                config_hash=RUN_ID,source_revision=GAT_PIN,input_hash=protocol['input_hash'])
            row = json.loads((run/'arms'/f'{mid}_s{seed}.json').read_text())
            marker = json.loads((run/'progress/stages'/f"{ident['stage']}_{mid}_s{seed}.completed.json").read_text())
            if (row.get('identity') != ident or row.get('model') != label
                    or row.get('seed') != seed or row.get('epochs') != 300
                    or [r.get('epoch') for r in row.get('training_history',[])] != list(range(1,301))
                    or len(row.get('metrics',{})) != 142
                    or not all(math.isfinite(v) for v in row['metrics'].values())
                    or marker.get('identity') != ident or marker.get('status') != 'completed'):
                raise RuntimeError(f'GAT seed{seed} {label} 정상 완료·저장 검증 실패')
    if '요청한 작업 완료.' not in log.read_text()[-8192:]:
        raise RuntimeError('GAT 정상 종료 로그 없음')


def status(phase, **extra):
    platform.atomic_json(ROOT/'queue_status.json',dict(phase=phase,
        updated_utc=datetime.now(timezone.utc).isoformat(),pod=POD,
        next_job='hm-ngcf',seed=50,models=list(MODELS),**extra))
    print(phase,extra,flush=True)


def queue(mode,pid):
    with platform.exclusive(core.SHARED/'locks/dh-gat-ada-to-ngcf50-m5.queue.lock'):
        try:
            core.preflight(ROOT,pod=POD)
            ticks = process_identity(pid)
            if ticks is None:
                completion_check()
            if mode == 'check':
                print('QUEUE_PREFLIGHT_OK',dict(pid=pid,waiting=ticks is not None,
                    models=MODELS,seed=50),flush=True)
                return
            status('waiting_for_dh_gat_ada_completion',prior_pid=pid,process_start_ticks=ticks)
            if ticks is not None:
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    fd = None
                if fd is not None:
                    try:
                        if process_identity(pid) not in (None,ticks):
                            raise RuntimeError('대기 PID 재사용')
                        poll = select.poll();poll.register(fd,select.POLLIN)
                        while not poll.poll(30_000):pass
                    finally:
                        os.close(fd)
            completion_check()
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU에 다른 계산 프로세스가 있어 자동 시작 차단')
            log = ROOT/'seed50_m5_run.log'
            with log.open('ab') as out:
                child = subprocess.Popen([str(core.SHARED/'venv/bin/python'),'-u',
                    str(Path(__file__).resolve()),'run'],stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                status('running_ngcf_seed50_m5',child_pid=child.pid,log=str(log))
                code = child.wait()
            if code:raise RuntimeError(f'NGCF50 M5 종료 코드{code}; 자동 재시도 없음')
            status('completed_ngcf_seed50_m5',log=str(log))
        except Exception as exc:
            status('failed_no_automatic_retry',error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','wait','run'))
    p.add_argument('--prior-pid',type=int,default=8880)
    args=p.parse_args()
    if args.mode=='run':
        completion_check()
        core.run_seed50(ROOT,models=MODELS,pod=POD)
    else:queue(args.mode,args.prior_pid)


if __name__=='__main__':main()
