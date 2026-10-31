"""Stop only the named pod after the current M5_D result is durably verified."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import zipfile

POD = 'xqsonboopgwkyo'
PID = 27141
START = '445766713'
ROOT = Path('/workspace/clv_hm_graphsage_seed49_m5_d')
RUN = ROOT / 'results/817e8e85d9ec'
LOG = Path('/workspace/hm_graphsage49_d_recovery_76d3756.log')
MID = 'graphsage_m5_d_lastweek'

def active():
    try:
        fields = Path(f'/proc/{PID}/stat').read_text().split()
        return fields[21] == START and fields[2] != 'Z'
    except FileNotFoundError:
        return False

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def status(phase, **extra):
    data = dict(phase=phase, pod=POD, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    path = ROOT / 'graphsage49_d_autostop_status.json'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    print(json.dumps(data), flush=True)

def verify():
    assert not active(), 'Training still active'
    assert 'COMPLETED seed49 M5_D' in LOG.read_text(), 'No completion log'
    protocol = json.loads((RUN/'protocol.json').read_text())
    arm = RUN/'arms'/f'{MID}_s49.json'
    row = json.loads(arm.read_text())
    assert row['model'] == 'M5_D' and row['seed'] == 49 and row['epochs'] == 300
    assert row['identity']['config_hash'] == RUN.name
    assert [r['epoch'] for r in row['training_history']] == list(range(1,301))
    assert len(row['metrics']) == 142 and all(math.isfinite(v) for v in row['metrics'].values())
    marker = json.loads((RUN/'progress/stages'/f"{protocol['code_version']}_{MID}_s49.completed.json").read_text())
    assert marker['status'] == 'completed' and marker['identity'] == row['identity']
    checkpoint = Path(marker['checkpoint_path'])
    assert checkpoint.parent == RUN/'progress/resume'
    assert checkpoint.stat().st_size == marker['checkpoint_bytes']
    assert digest(checkpoint) == marker['checkpoint_sha256']
    # Single-arm exports intentionally have a _partial suffix (not all eight arms).
    archives = [p for p in (ROOT/'hm_graphsage_seed49_m5_d_results.zip', ROOT/'hm_graphsage_seed49_m5_d_results_partial.zip') if p.is_file()]
    assert len(archives) == 1, 'Missing or ambiguous result ZIP'
    archive = archives[0]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read('arms/'+arm.name) == arm.read_bytes()
        assert z.read('protocol.json') == (RUN/'protocol.json').read_bytes()
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True, timeout=30)
    assert not gpu.strip(), 'Other GPU work remains'
    return dict(zip=str(archive), zip_sha256=digest(archive), checkpoint_sha256=marker['checkpoint_sha256'])

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    assert os.environ.get('RUNPOD_POD_ID') == POD
    assert os.environ.get('RUNPOD_API_KEY'), 'No pod stop credential'
    assert Path('/usr/bin/runpodctl').is_file()
    if args.check:
        print(json.dumps(dict(pod=POD, training_active=active(), stop_now=False, mode='check_only'))); return
    with (ROOT/'graphsage49_d_autostop.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        status('armed_waiting_for_training', training_pid=PID)
        while active(): time.sleep(60)
        for attempt in range(1,6):
            try:
                evidence = verify(); status('verified_ready_to_stop', **evidence); os.sync()
                result = subprocess.run(['/usr/bin/runpodctl','stop','pod',POD], capture_output=True, text=True, timeout=90)
                if result.returncode == 0:
                    status('stop_request_accepted', **evidence); return
                status('stop_request_failed', attempt=attempt, returncode=result.returncode)
            except Exception as exc:
                status('verification_failed_no_stop', attempt=attempt, error=f'{type(exc).__name__}: {exc}')
            time.sleep(60)
        status('needs_attention_no_stop')

if __name__ == '__main__': main()
