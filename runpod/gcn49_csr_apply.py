"""Opt-in execution-layout change for the existing H&M GCN49 run only."""
import sys, os, json, time, signal, shutil, hashlib, tempfile
from pathlib import Path
REPO = Path('/workspace/gcn49_followup_08fac20')
ROOT = Path('/workspace/clv_hm_gcn_seed49_all8')
sys.path[:0] = [str(REPO), str(REPO/'runpod')]

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

def validate():
    import torch, copy
    import clv_dh_gcn_final8_seed49 as s
    from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as ref
    from test_clv_dh_gcn_final8_seed49 import prepared as fixture, test_resume_restores_optimizer_and_rng_exactly
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='gcn-csr-check-') as d:
        p = fixture.__wrapped__(ref.__wrapped__(Path(d)))
        original = s.build_model
        for label in s.ORDER:
            s.base.v3.set_seed(49)
            a,w = original(p,label); a=a.cpu(); b=convert(copy.deepcopy(a))
            u=torch.tensor([0,1,2]); pos=torch.tensor([1,2,3]); neg=torch.tensor([[4],[4],[0]])
            weights=None if w is None else torch.tensor(w[:3],dtype=torch.float32)
            oa=torch.optim.Adam(a.parameters(),lr=.0005); ob=torch.optim.Adam(b.parameters(),lr=.0005)
            for step in range(3):
                oa.zero_grad(); ob.zero_grad()
                la=s.base.fixed.recheck._batch_loss(a,u,pos,neg,weights)[0]
                lb=s.base.fixed.recheck._batch_loss(b,u,pos,neg,weights)[0]
                la.backward(); lb.backward()
                torch.testing.assert_close(la,lb,atol=2e-6,rtol=2e-4)
                for (n,x),(m,y) in zip(a.named_parameters(),b.named_parameters()):
                    assert n==m
                    torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-4)
                oa.step(); ob.step()
            for k,v in a.state_dict().items():
                torch.testing.assert_close(v,b.state_dict()[k],atol=2e-6,rtol=2e-4)
            b.load_state_dict(a.state_dict(),strict=True); ob.load_state_dict(oa.state_dict())
            print('PARITY_AND_STATE_PASS',label,flush=True)
        def build(p,label):
            m,w=original(p,label); return convert(m),w
        s.build_model=build
        test_resume_restores_optimizer_and_rng_exactly(p,Path(d)/'resume_test')
        print('CSR_SAVE_RESUME_EXACT_PASS',flush=True)

def apply(pid):
    import torch
    assert os.environ.get('RUNPOD_POD_ID')=='rkdcwyt2m28a1b'
    expected=['/workspace/clv/venv/bin/python','-u',str(REPO/'runpod/after_ngcf50_then_gcn49.py'),'run']
    assert Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')==expected
    run=ROOT/'results/hm-gcn/b4079e349e33'
    # Do not interrupt final test evaluation or an already completed arm.
    assert not (run/'arms/gcn_m5_a_lastweek_s49.json').exists()
    latest=run/'progress/resume/clv-hm-gcn-final8-seed49-v1_gcn_m5_a_lastweek_s49_latest.pt'
    backup=ROOT/'diagnostics'/('csr_apply_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    backup.mkdir(parents=True,exist_ok=False)
    terminated=False
    os.kill(pid,signal.SIGSTOP)
    try:
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        epoch=int(ck['epoch']); assert 1<=epoch<295
        assert 'optimizer_state' in ck and 'torch_rng_state' in ck and 'numpy_rng_state' in ck
        target=backup/latest.name; shutil.copy2(latest,target)
        def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha(latest)==sha(target)
        receipt=dict(seed=49,old_pid=pid,resume_epoch=epoch,checkpoint=str(latest),backup=str(target),sha256=sha(target),change='COO -> CSR only; same graph values, FP32, batch, optimizer, epoch and test protocol',seed50_untouched=True)
        (backup/'receipt.json').write_text(json.dumps(receipt,indent=2))
        os.kill(pid,signal.SIGTERM); terminated=True
    finally:
        os.kill(pid,signal.SIGCONT)
    for _ in range(60):
        proc=Path(f'/proc/{pid}/stat')
        if not proc.exists() or proc.read_text().rsplit(') ',1)[1].split()[0]=='Z': break
        time.sleep(1)
    else: raise RuntimeError('Original process has not exited; refusing duplicate training')
    resume(backup/'receipt.json')

def resume(receipt_path):
    assert os.environ.get('RUNPOD_POD_ID')=='rkdcwyt2m28a1b'
    receipt_path=Path(receipt_path)
    receipt=json.loads(receipt_path.read_text())
    assert receipt['seed']==49 and receipt['seed50_untouched']
    assert hashlib.sha256(Path(receipt['backup']).read_bytes()).hexdigest()==receipt['sha256']
    import after_ngcf50_then_gcn49 as launch
    def status(phase,**kw):
        path=ROOT/'csr_queue_status.json'
        tmp=path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(dict(phase=phase,**kw),indent=2))
        tmp.replace(path)
    launch.status=status
    original_load=launch.platform.load_runner
    def load(job,root):
        screen,cfg=original_load(job,root)
        assert cfg.seeds==(49,) and cfg.epochs==300
        original_build=screen.build_model
        def build(p,label):
            model,w=original_build(p,label)
            convert(model)
            print('CSR_APPLIED',label,'same nnz',model.adj._nnz(),flush=True)
            return model,w
        screen.build_model=build
        return screen,cfg
    launch.platform.load_runner=load
    launch.status('running_gcn_seed49_csr',child_pid=os.getpid(),log=str(ROOT/'gcn_seed49_csr_run.log'),receipt=str(receipt_path))
    launch.run()
    launch.status('completed_gcn_seed49_csr',receipt=str(receipt_path))

if __name__=='__main__':
    if sys.argv[1]=='validate': validate()
    elif sys.argv[1]=='apply': apply(int(sys.argv[2]))
    elif sys.argv[1]=='resume': resume(sys.argv[2])
    else: raise ValueError('validate or apply only')

