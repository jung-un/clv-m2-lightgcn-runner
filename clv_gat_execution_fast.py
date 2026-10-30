"""Opt-in GAT execution patch, not a new scientific configuration.

Same FP32 full graph/attention/loss/Adam and NumPy draws. CSR changes reduction
order (roundoff, not bitwise equivalence). No learned embedding is cached.
Indices are ephemeral; state_dict and original run/checkpoint identities stay
unchanged. Install only in an idle runtime; provenance is stored separately.
"""
import torch
from torch.nn import functional as F
from types import MethodType
import numpy as np

class GraphIndex:
    def __init__(self, receiver, source, n):
        if receiver.ndim != 1 or receiver.shape != source.shape:
            raise ValueError('invalid graph index shape')
        if len(receiver) and ((receiver[1:] < receiver[:-1]).any()
                or receiver.min() < 0 or source.min() < 0
                or receiver.max() >= n or source.max() >= n):
            raise ValueError('requires sorted valid COO indices')
        self.r,self.s,self.n=receiver,source,n
        self.crow=torch.cat((receiver.new_zeros(1),torch.bincount(receiver,minlength=n).cumsum(0)))
        self.perm=torch.argsort(source,stable=True)
        self.tc=torch.cat((source.new_zeros(1),torch.bincount(source,minlength=n).cumsum(0)))
        self.ts=receiver[self.perm]
    def matrix(self,w,transpose=False):
        if transpose:
            return torch.sparse_csr_tensor(self.tc,self.ts,w[self.perm],size=(self.n,self.n))
        return torch.sparse_csr_tensor(self.crow,self.s,w,size=(self.n,self.n))

class Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g):
        ctx.save_for_backward(x,w);ctx.g=g
        return torch.sparse.mm(g.matrix(w),x)
    @staticmethod
    def backward(ctx,dy):
        x,w=ctx.saved_tensors;g=ctx.g
        dx=torch.sparse.mm(g.matrix(w,True),dy) if ctx.needs_input_grad[0] else None
        dw=torch.empty_like(w) if ctx.needs_input_grad[1] else None
        if dw is not None:
            for a in range(0,len(w),262144):
                b=min(a+262144,len(w))
                dw[a:b]=(dy.index_select(0,g.r[a:b])*x.index_select(0,g.s[a:b])).sum(1)
        return dx,dw,None

def apply(model):
    if getattr(model, '_gat_execution_fast', False):
        return model
    graph=GraphIndex(model.edge_receiver,model.edge_source,model.n_users+model.n_items)
    def forward(self,features,receiver,source,n_nodes,log_prior):
        weights,x=self.attention_weights(features,receiver,source,n_nodes,log_prior)
        return F.elu(Sum.apply(x,weights,graph))
    for layer in model.gat_layers:
        layer.forward=MethodType(forward,layer)
    model._gat_execution_fast = True
    return model

def sample(users,positives,n_items,positive_keys,rng,*,k):
    if k!=1:raise ValueError('uniform K1 only')
    neg=rng.integers(0,n_items,size=len(users))
    offset=users.astype(np.int64)*n_items
    pending=np.arange(len(users))
    for attempt in range(51):
        key=offset[pending]+neg[pending]
        pos=np.searchsorted(positive_keys,key).clip(max=len(positive_keys)-1)
        pending=pending[positive_keys[pos]==key]
        if not len(pending):return neg[:,None]
        if attempt==50:raise RuntimeError('observed negative remains')
        neg[pending]=rng.integers(0,n_items,size=len(pending))


def install(screen, prepared):
    """Patch only this runner. Retain old latest as previous after successful save."""
    from pathlib import Path
    import hashlib, json, os, shutil, tempfile
    root = Path(prepared['run_dir']).resolve()
    if getattr(screen, '_gat_execution_fast_root', None) == str(root):
        return
    if getattr(screen, '_gat_execution_fast_root', None):
        raise RuntimeError('another run already patched')
    if prepared['cfg'].epochs != 300 or prepared['cfg'].seeds != (50,):
        raise RuntimeError('this opt-in patch is for fixed seed50/300epoch')
    if tuple(screen.ORDER) not in (('M1', 'M2', 'M3', 'M4'),
                                  ('M5_A', 'M5_B', 'M5_C', 'M5_D')):
        raise RuntimeError('unexpected runner')
    original_build = screen.build_model
    def build(p, label):
        if Path(p['run_dir']).resolve() != root:
            raise RuntimeError('different run')
        model, weights = original_build(p, label)
        return apply(model), weights

    original_store = screen.SafeProgressStore
    class Store(original_store):
        def save_epoch(self, model, optimizer, rng, **state):
            if self.root.resolve() != root / 'progress':
                raise RuntimeError('different checkpoint root')
            # Keep all old files on failure. No Drive API listing/deletion.
            folder = tempfile.mkdtemp(prefix='gat-previous-')
            try:
                old = Path(folder) / 'previous.pt'
                if self.latest_checkpoint.exists():
                    shutil.copyfile(self.latest_checkpoint, old)
                result = super().save_epoch(model, optimizer, rng, **state)
                if old.exists():
                    previous = self.latest_checkpoint.with_name(
                        self.latest_checkpoint.name.replace('_latest.pt', '_previous.pt'))
                    if previous == self.latest_checkpoint:
                        raise RuntimeError('unexpected checkpoint filename')
                    temporary = previous.with_suffix('.pt.tmp')
                    digest = hashlib.sha256()
                    with old.open('rb') as source, temporary.open('wb') as target:
                        for block in iter(lambda: source.read(8 * 1024**2), b''):
                            digest.update(block); target.write(block)
                        target.flush(); os.fsync(target.fileno())
                    verify = hashlib.sha256()
                    with temporary.open('rb') as source:
                        for block in iter(lambda: source.read(8 * 1024**2), b''):
                            verify.update(block)
                    if verify.digest() != digest.digest():
                        raise IOError('previous copy mismatch; files preserved')
                    os.replace(temporary, previous)
            except BaseException:
                # Even a metadata failure AFTER latest was published must not
                # discard the only copy of its predecessor.
                print(f'GAT save interrupted: recovery copy retained at {folder}', flush=True)
                raise
            else:
                shutil.rmtree(folder)
                return result
    # No original source/protocol/checkpoint identity is overwritten.
    receipt = dict(version='gat-csr-execution-v2', run_dir=str(root), models=list(screen.ORDER),
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes=['CSR neighbor sum; exact analytic gradients',
                 'same uniform RNG draws; inspect only unresolved negatives',
                 'preserve previous checkpoint after successful latest save'],
        unchanged=['FP32', 'all edges', 'batch', 'loss', 'Adam', 'seed', '300epoch',
                   'test protocol', 'state_dict keys', 'checkpoint identity'],
        numerical_note='FP32 reduction order differs; not bitwise equivalence')
    target = root / 'gat_execution_optimization.json'
    if target.exists() and json.loads(target.read_text()) != receipt:
        raise RuntimeError('different execution receipt already present')
    if not target.exists():
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt, indent=2)); os.replace(temp, target)
    screen.build_model = build
    screen.sample_uniform_negative_matrix = sample
    screen.SafeProgressStore = Store
    screen._gat_execution_fast_root = str(root)
    print('GAT execution optimization installed: scientific settings unchanged; latest/previous retained', flush=True)
