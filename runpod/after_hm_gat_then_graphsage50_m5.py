"""Existing GAT49 core completion -> independent GraphSAGE50 M5 partition."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys

import launch as platform
import hm_ngcf_m5 as storage

POD = 'xqsonboopgwkyo'
ROOT = Path('/workspace/clv_hm_graphsage_seed50_m5')
SHARED = Path('/workspace/clv')
PRIOR = SHARED / 'results/hm-gat/650f38949cf8'
MODELS = ('M5_A', 'M5_B', 'M5_C', 'M5_D')
PRIOR_ARGS = ['-u', '/workspace/hm_gat_start/launch.py', 'run', '--job', 'hm-gat',
              '--seeds', '49', '--models', 'M1', 'M2', 'M3', 'M4', '--confirmed-unstarted']

def process_identity(pid):
    try:
        p = Path('/proc') / str(pid)
        args = (p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        ticks = (p/'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None
    if args[1:] != PRIOR_ARGS:
        raise RuntimeError('대상 PID가 승인된 H&M GAT49 M1~M4 실행과 다릅니다')
    return ticks

def completion_check(prior=PRIOR):
    p = json.loads((prior/'protocol.json').read_text())
    if (p['source_revision'] != platform.JOBS['hm-gat']['pin']
            or p['config']['epochs'] != 300 or p['config']['seeds'] != [49]
            or any(p[k] for k in ('validation','holdout','early_stopping'))):
        raise RuntimeError('GAT 선행 실행 조건 불일치')
    for label in ('M1','M2','M3','M4'):
        mid = f'gat_{label.lower()}_lastweek'
        ident = dict(stage=p['code_version'], model_id=mid, seed=49,
                     config_hash=prior.name, source_revision=p['source_revision'], input_hash=p['input_hash'])
        r = json.loads((prior/'arms'/f'{mid}_s49.json').read_text())
        m = json.loads((prior/'progress/stages'/f"{p['code_version']}_{mid}_s49.completed.json").read_text())
        if (r.get('identity') != ident or r.get('epochs') != 300 or r.get('seed') != 49
                or r.get('model') != label or len(r.get('metrics',{})) != 142
                or not all(math.isfinite(v) for v in r['metrics'].values())
                or [h['epoch'] for h in r.get('training_history',[])] != list(range(1,301))
                or m.get('identity') != ident or m.get('status') != 'completed'):
            raise RuntimeError('선행 모형 완료 검증 실패: '+label)

def preflight():
    if os.environ.get('RUNPOD_POD_ID') != POD:
        raise RuntimeError('예약된 hm-gat-seed49 팟에서만 실행합니다')
    ROOT.mkdir(parents=True, exist_ok=True)
    platform.verify_files('hm-gat', SHARED)
    platform.guard_space(ROOT)
    repo = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise RuntimeError('전용 실행 소스에 미확정 변경이 있습니다')
    return repo, revision

def run():
    completion_check()
    repo, revision = preflight()
    with platform.exclusive(SHARED/'locks/hm-graphsage-seed50-m5.lock'):
        sys.path.insert(0,str(repo))
        import torch
        platform.cuda_check(torch)
        screen = importlib.import_module('clv_hm_graphsage_m5_seed50')
        data = SHARED/'data'
        screen.base.v3.SCHEMA['hm'].update(tx_path=str(data/'raw/hm/transactions_train.parquet'),
                                          item_meta_path=str(data/'raw/hm/articles.csv'))
        cfg = screen.configure(out_dir=str(ROOT/'results'),
            m2_reference=str(data/'references/clv_m2_adaptive_hm_seed49_lastweek_results.zip'),
            m4_reference=str(data/'references/clv_hm_m4c_lastweek_seed49_results.zip'))
        assert screen.ORDER == MODELS and cfg.seeds == (50,) and cfg.epochs == 300
        storage.install_storage(screen, ROOT)
        screen.SafeProgressStore._runpod_storage_installed = True
        env = dict(source_commit=revision, pod=POD, seed=50, models=list(MODELS),
                   torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                   tf32=torch.backends.cuda.matmul.allow_tf32,
                   checkpoint='full model/Adam/RNG each epoch; atomic latest and previous on network volume')
        platform.bind_environment(ROOT/'environment.json',env)
        prepared = screen.prepare(cfg)
        assert set(prepared['data']['splits']) == {'test'}
        for label in MODELS:
            with platform.evaluation_once(screen,prepared,label,50):
                screen.run_arm(prepared,label)
            screen.export_zip(prepared,str(ROOT/'hm_graphsage_seed50_m5_results.zip'))
        assert screen.report(prepared)['complete']
        print('GRAPHSAGE50_M5_COMPLETE; Pod 과금은 계속됩니다.',flush=True)

def status(phase, **extra):
    platform.atomic_json(ROOT/'queue_status.json',dict(phase=phase,pod=POD,
        next_job='hm-graphsage',seed=50,models=list(MODELS),
        updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    print(phase,extra,flush=True)

def queue(mode,pid):
    with platform.exclusive(SHARED/'locks/hm-gat49-to-graphsage50-m5.queue.lock'):
        try:
            preflight()
            ticks = process_identity(pid)
            if ticks is None:
                completion_check()
            if mode == 'check':
                print('QUEUE_PREFLIGHT_OK',dict(prior_pid=pid,waiting=ticks is not None,models=MODELS),flush=True)
                return
            status('waiting_for_gat49_core_completion',prior_pid=pid,process_start_ticks=ticks)
            if ticks is not None:
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    fd = None
                if fd is not None:
                    try:
                        if process_identity(pid) not in (None,ticks):
                            raise RuntimeError('PID 재사용 감지')
                        poll=select.poll(); poll.register(fd,select.POLLIN)
                        while not poll.poll(30000):
                            pass
                    finally:
                        os.close(fd)
            completion_check()
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU의 다른 작업이 남아 있습니다')
            log=ROOT/'seed50_m5_run.log'
            with log.open('ab') as out:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],
                    stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                status('running_graphsage50_m5',child_pid=child.pid,log=str(log))
                code=child.wait()
            if code:
                raise RuntimeError(f'후속 작업 종료코드 {code}; 자동 재시도 없음')
            status('completed_graphsage50_m5',log=str(log))
        except Exception as exc:
            status('failed_no_automatic_retry',error=str(exc))
            raise

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','wait','run'))
    parser.add_argument('--prior-pid',type=int,default=1945)
    args=parser.parse_args()
    run() if args.mode=='run' else queue(args.mode,args.prior_pid)
