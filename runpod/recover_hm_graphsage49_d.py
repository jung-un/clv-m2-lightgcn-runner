"""Wait for verified GraphSAGE seed50 M5 completion; run only seed49 M5_D."""
import fcntl, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

SOURCE = Path('/workspace/hm_graphsage49_d_followup_76d3756')
OUT = Path('/workspace/clv_hm_graphsage_seed49_m5_d')
PRIOR = Path('/workspace/clv_hm_graphsage_seed50_m5/results/54bf0390d53b')
PRIOR_PID = 22921
PIN = '76d3756959bd4350c19d9ecc3ad7e5af0df7de20'
def check_prior():
    protocol = json.loads((PRIOR/'protocol.json').read_text())
    assert protocol['config']['seeds'] == [50] and protocol['config']['epochs'] == 300
    for label in ('M5_A','M5_B','M5_C','M5_D'):
        mid = 'graphsage_'+label.lower()+'_lastweek'
        row = json.loads((PRIOR/'arms'/f'{mid}_s50.json').read_text())
        marker = json.loads((PRIOR/'progress/stages'/f"{protocol['code_version']}_{mid}_s50.completed.json").read_text())
        assert row['model']==label and row['seed']==50 and row['epochs']==300
        assert row['identity']==marker['identity'] and marker['status']=='completed'
        assert [r['epoch'] for r in row['training_history']]==list(range(1,301))
        assert len(row['metrics'])==142 and all(math.isfinite(x) for x in row['metrics'].values())
    return protocol
def alive():
    try: return (Path('/proc')/str(PRIOR_PID)/'cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError: return None
def main():
    assert os.environ.get('RUNPOD_POD_ID')=='xqsonboopgwkyo'
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==PIN
    if alive() is not None:
        assert '/workspace/graphsage50_m5_followup/runpod/after_hm_gat_then_graphsage50_m5.py' in alive()
    OUT.mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('WAITING seed50 M5_D; next ONLY seed49 M5_D',flush=True)
        while alive() is not None:
            if '/workspace/graphsage50_m5_followup/runpod/after_hm_gat_then_graphsage50_m5.py' not in alive(): raise RuntimeError('PID 재사용 감지')
            time.sleep(30)
        previous=check_prior()
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if gpu: raise RuntimeError('GPU에 기존 작업이 남아 있어 시작하지 않음')
        if any(OUT.glob('results/*/evaluation_started/M5_D_s49.json')): raise RuntimeError('기존 평가 시작 기록 확인 필요')
        sys.path.insert(0,str(SOURCE));sys.path.insert(1,'/workspace/graphsage50_m5_followup/runpod')
        import torch, clv_hm_graphsage_final8_seed49 as screen
        import hm_ngcf_m5 as storage
        assert torch.cuda.is_available() and os.statvfs('/workspace').f_bavail*os.statvfs('/workspace').f_frsize > 8_000_000_000
        data=Path('/workspace/clv/data');screen.base.v3.SCHEMA['hm'].update(tx_path=str(data/'raw/hm/transactions_train.parquet'),item_meta_path=str(data/'raw/hm/articles.csv'))
        cfg=screen.configure(out_dir=str(OUT/'results'),m2_reference=str(data/'references/clv_m2_adaptive_hm_seed49_lastweek_results.zip'),m4_reference=str(data/'references/clv_hm_m4c_lastweek_seed49_results.zip'))
        assert cfg.seeds==(49,) and cfg.epochs==300 and screen.ARMS['M5_D']==(True,True,True)
        storage.install_storage(screen,OUT)
        prepared=screen.prepare(cfg);protocol=prepared['protocol']
        assert prepared['input_hash']==previous['input_hash'] and set(prepared['data']['splits'])=={'test'}
        assert protocol['data_stats']['min_item_inter']==1 and not any(protocol[k] for k in ('validation','holdout','early_stopping'))
        assert protocol['data_stats']['split_boundaries']['test']['end_inclusive'].startswith('2020-09-22')
        import launch as platform
        print('START seed49 M5_D only',flush=True)
        with platform.evaluation_once(screen,prepared,'M5_D',49): row=screen.run_arm(prepared,'M5_D')
        assert row['epochs']==300 and row['seed']==49 and row['model']=='M5_D'
        print('COMPLETED seed49 M5_D',flush=True)
        screen.export_zip(prepared,str(OUT/'hm_graphsage_seed49_m5_d_results.zip'))
if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc();sys.exit(1)
