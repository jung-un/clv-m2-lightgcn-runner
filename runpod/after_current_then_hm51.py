"""Two explicitly assigned full-eight seed51 jobs; wait for validated prior output."""
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
import zipfile
import launch as platform
import hm_ngcf_m5 as storage

SHARED = Path('/workspace/clv')
MODELS = ('M1','M2','M3','M4','M5_A','M5_B','M5_C','M5_D')
JOBS = {
 'gat': dict(pod='xqsonboopgwkyo', prior='/workspace/clv_hm_graphsage_seed49_m5_d',
             run='results/817e8e85d9ec', backbone='graphsage', labels=('M5_D',), process='hm_graphsage49_d_recovery',
             root='/workspace/clv_hm_gat_seed51_all8'),
 'graphsage': dict(pod='rkdcwyt2m28a1b', prior='/workspace/clv_hm_gcn_seed49_all8',
             run='results/hm-gcn/b4079e349e33', backbone='gcn', labels=MODELS, process='gcn49_csr_resume.py',
             root='/workspace/clv_hm_graphsage_seed51_all8')}

def process_identity(pid, job):
    try:
        p=Path('/proc')/str(pid)
        cmd=(p/'cmdline').read_bytes().decode().replace('\0',' ')
        ticks=(p/'stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:
        return None
    if JOBS[job]['process'] not in cmd:
        raise RuntimeError('선행 PID의 실행 코드 불일치: 중복 학습 방지')
    return ticks

def completion_check(job):
    spec=JOBS[job]; root=Path(spec['prior'])
    run=root/spec['run']; p=json.loads((run/'protocol.json').read_text())
    assert p['config']['epochs']==300 and p['config']['seeds']==[49]
    assert not any(p.get(k,False) for k in ('validation','holdout','early_stopping'))
    for label in spec['labels']:
        mid=f"{spec['backbone']}_{label.lower()}_lastweek"
        ident=dict(stage=p['code_version'],model_id=mid,seed=49,config_hash=run.name,
                   source_revision=p['source_revision'],input_hash=p['input_hash'])
        r=json.loads((run/'arms'/f'{mid}_s49.json').read_text())
        marker=run/'progress/stages'/f"{p['code_version']}_{mid}_s49.completed.json"
        m=json.loads(marker.read_text())
        assert r['identity']==ident and r['seed']==49 and r['model']==label and r['epochs']==300
        assert len(r['metrics'])==142 and all(math.isfinite(v) for v in r['metrics'].values())
        assert [h['epoch'] for h in r['training_history']]==list(range(1,301))
        assert m['identity']==ident and m['status']=='completed'
    # Ensure a readable results packet already exists before consuming the GPU.
    archives=list(root.glob('*results*.zip'))+list((root/'exports').glob('*results*.zip'))
    if not archives:
        raise RuntimeError('선행 ZIP 저장 미확인')
    with zipfile.ZipFile(max(archives,key=lambda x:x.stat().st_mtime)) as z:
        assert z.testzip() is None and z.namelist()

def preflight(job):
    spec=JOBS[job]; root=Path(spec['root'])
    assert os.environ.get('RUNPOD_POD_ID')==spec['pod'], '지정 팟 이외 실행 금지'
    root.mkdir(parents=True,exist_ok=True)
    platform.verify_files('hm-gat',SHARED)
    platform.guard_space(root)
    repo=Path(__file__).resolve().parent.parent
    rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    return root,repo,rev

def status(job,phase,**extra):
    root=Path(JOBS[job]['root'])
    value=dict(phase=phase,pod=JOBS[job]['pod'],next_job=job,seed=51,models=list(MODELS),
               updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
    platform.atomic_json(root/'queue_status.json',value)
    print(json.dumps(value),flush=True)

def run(job):
    completion_check(job)
    root,repo,rev=preflight(job)
    with platform.exclusive(SHARED/f'locks/hm-{job}-seed51-all8.lock'):
        sys.path.insert(0,str(repo))
        import torch
        platform.cuda_check(torch)
        screen=importlib.import_module(f'clv_hm_{job}_final8_seed51')
        data=SHARED/'data'
        screen.base.v3.SCHEMA['hm'].update(tx_path=str(data/'raw/hm/transactions_train.parquet'),
              item_meta_path=str(data/'raw/hm/articles.csv'))
        cfg=screen.configure(out_dir=str(root/'results'),
              m2_reference=str(data/'references/clv_m2_adaptive_hm_seed49_lastweek_results.zip'),
              m4_reference=str(data/'references/clv_hm_m4c_lastweek_seed49_results.zip'))
        assert cfg.seeds==(51,) and cfg.epochs==300 and screen.ORDER==MODELS
        storage.install_storage(screen,root)
        platform.bind_environment(root/'environment.json',dict(source_commit=rev,pod=JOBS[job]['pod'],
             seed=51,models=list(MODELS),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
             tf32=torch.backends.cuda.matmul.allow_tf32,
             optimization='fixed input once; atomic latest/previous every epoch; GraphSAGE CSR, GAT unchanged'))
        prepared=screen.prepare(cfg)
        assert set(prepared['data']['splits'])=={'test'}
        for label in MODELS:
            with platform.evaluation_once(screen,prepared,label,51):
                screen.run_arm(prepared,label)
            screen.export_zip(prepared,str(root/f'hm_{job}_seed51_all8_results.zip'))
        assert screen.report(prepared)['complete']

def queue(job,mode,pid):
    with platform.exclusive(SHARED/f'locks/hm-{job}-seed51-all8.queue.lock'):
        try:
            root,repo,_=preflight(job)
            ticks=process_identity(pid,job)
            if ticks is None: completion_check(job)
            if mode=='check':
                print('QUEUE_PREFLIGHT_OK',job,pid,ticks,flush=True); return
            status(job,'waiting_for_prior_completion',prior_pid=pid,process_start_ticks=ticks)
            if ticks is not None:
                try: fd=os.pidfd_open(pid)
                except ProcessLookupError: fd=None
                if fd is not None:
                    try:
                        assert process_identity(pid,job) in (None,ticks)
                        poll=select.poll(); poll.register(fd,select.POLLIN)
                        while not poll.poll(30000): pass
                    finally: os.close(fd)
            completion_check(job)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), '다른 GPU 작업 실행 중'
            log=root/'seed51_all8_run.log'
            with log.open('ab') as out:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),job,'run'],
                    cwd=repo,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                status(job,'running_seed51_all8',child_pid=child.pid,log=str(log))
                code=child.wait()
            assert code==0, f'후속 실행 실패: {code}'
            status(job,'completed_seed51_all8',log=str(log))
        except Exception as exc:
            status(job,'failed_no_automatic_retry',error=str(exc)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('job',choices=JOBS)
    p.add_argument('mode',choices=('check','wait','run')); p.add_argument('--prior-pid',type=int)
    a=p.parse_args()
    run(a.job) if a.mode=='run' else queue(a.job,a.mode,a.prior_pid)
