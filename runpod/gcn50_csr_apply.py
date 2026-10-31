"""Resume the existing H&M GCN50 run with CSR storage, not new settings."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path('/workspace/gcn50_followup_c896a68')
ROOT = Path('/workspace/clv_hm_gcn_seed50_all8')
POD = '4mgakh8utyoyqu'
RUN = ROOT / 'results/hm-gcn/58dcc698dde3'
sys.path[:0] = [str(REPO), str(REPO / 'runpod')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def convert(model):
    import torch
    old = model.adj
    assert old.layout == torch.sparse_coo and old.is_coalesced()
    keys = tuple(model.state_dict())
    csr = old.to_sparse_csr()
    assert torch.equal(csr.values(), old.values())
    assert torch.equal(csr.col_indices(), old.indices()[1])
    assert csr.shape == old.shape and csr._nnz() == old._nnz()
    model.adj = csr
    assert tuple(model.state_dict()) == keys
    return model


def resume(receipt_path):
    assert os.environ.get('RUNPOD_POD_ID') == POD
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt['seed'] == 50 and receipt['run'] == str(RUN)
    assert sha(receipt['backup']) == receipt['sha256']
    import after_gat49_m5_then_gcn50 as launch

    def status(phase, **kw):
        # The old queue parent can still report its interrupted child's exit.
        path = ROOT / 'csr_queue_status.json'
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(dict(phase=phase, seed=50, pod=POD,
                                      updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw), indent=2))
        tmp.replace(path)

    launch.status = status
    original_load = launch.platform.load_runner

    def load(job, root):
        screen, cfg = original_load(job, root)
        assert cfg.seeds == (50,) and cfg.epochs == 300
        original_build = screen.build_model

        def build(p, label):
            assert str(p['run_dir']) == str(RUN)
            model, weights = original_build(p, label)
            convert(model)
            print('CSR_APPLIED', label, 'same nnz', model.adj._nnz(), flush=True)
            return model, weights

        screen.build_model = build
        return screen, cfg

    launch.platform.load_runner = load
    status('running_gcn_seed50_csr', child_pid=os.getpid(),
           log=str(ROOT / 'gcn_seed50_csr_run.log'), receipt=str(receipt_path))
    try:
        launch.run()  # original lock, fixed protocol, storage and evaluation guards
    except BaseException as exc:
        status('failed_gcn_seed50_csr', error=repr(exc), receipt=str(receipt_path))
        raise
    status('completed_gcn_seed50_csr', receipt=str(receipt_path))


def apply(pid):
    import torch
    assert os.environ.get('RUNPOD_POD_ID') == POD
    proc = Path(f'/proc/{pid}')
    expected = ['/workspace/clv/venv/bin/python', '-u',
                str(REPO / 'runpod/after_gat49_m5_then_gcn50.py'), 'run']
    assert (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0') == expected
    latest = RUN / 'progress/resume/clv-hm-gcn-final8-seed50-v1_gcn_m1_lastweek_s50_latest.pt'
    assert not (RUN / 'arms/gcn_m1_lastweek_s50.json').exists()
    backup = ROOT / 'diagnostics' / ('csr_apply_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    backup.mkdir(parents=True, exist_ok=False)
    # Fail-safe: release a suspended old process even if this helper crashes.
    ticks = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    watchdog = "import os,time,pathlib,signal; time.sleep(120); p=pathlib.Path('/proc/%d/stat'); p.exists() and p.read_text().rsplit(') ',1)[1].split()[19]==%r and os.kill(%d,signal.SIGCONT)" % (pid, ticks, pid)
    subprocess.Popen([sys.executable, '-c', watchdog], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.kill(pid, signal.SIGSTOP)
    try:
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        epoch = int(ck['epoch'])
        assert 1 <= epoch < 290
        ident = ck['identity']
        assert ident['seed'] == 50 and ident['model_id'] == 'gcn_m1_lastweek'
        assert ident['config_hash'] == RUN.name
        assert [r['epoch'] for r in ck['history']] == list(range(1, epoch + 1))
        assert all(k in ck for k in ('optimizer_state', 'torch_rng_state', 'numpy_rng_state'))
        target = backup / latest.name
        shutil.copy2(latest, target)
        digest = sha(target)
        assert sha(latest) == digest
        receipt = dict(seed=50, run=str(RUN), old_pid=pid, resume_epoch=epoch,
                       checkpoint=str(latest), backup=str(target), sha256=digest,
                       change='COO -> CSR only; same graph, FP32, batch, optimizer and test protocol')
        receipt_path = backup / 'receipt.json'
        receipt_path.write_text(json.dumps(receipt, indent=2))
        print('BACKUP_VERIFIED', json.dumps(receipt), flush=True)
        os.kill(pid, signal.SIGTERM)
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    for _ in range(60):
        stat = proc / 'stat'
        if not stat.exists() or stat.read_text().rsplit(') ', 1)[1].split()[0] == 'Z':
            break
        time.sleep(1)
    else:
        raise RuntimeError('Old process remains; refusing duplicate training')
    resume(receipt_path)


if __name__ == '__main__':
    if sys.argv[1] == 'apply':
        apply(int(sys.argv[2]))
    elif sys.argv[1] == 'resume':
        resume(sys.argv[2])
    else:
        raise ValueError('Expected apply PID or resume RECEIPT')
