"""Overlap NumPy negative preparation with GCN compute; preserve epoch RNG boundary."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect


class Batches:
    def __init__(self, permutation, batch_size, tr_u, tr_i, n_items, keys, rng, sampler):
        self.args = permutation, batch_size, tr_u, tr_i, n_items, keys, rng, sampler
        self.count = (len(permutation) + batch_size - 1) // batch_size

    def _one(self, batch):
        permutation, size, tr_u, tr_i, n_items, keys, rng, sampler = self.args
        index = permutation[batch * size:(batch + 1) * size]
        users, positives = tr_u[index], tr_i[index]
        return index, sampler(users, positives, n_items, keys, rng, k=1)

    def __enter__(self):
        # Exactly one worker owns this epoch's NumPy RNG. No CUDA calls here.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gcn-negative')
        self.pending = self.pool.submit(self._one, 0)
        self.next = 0
        return self

    def take(self, batch):
        if batch != self.next:
            raise RuntimeError('Prefetch batch order changed')
        value = self.pending.result()
        self.next += 1
        if self.next < self.count:
            self.pending = self.pool.submit(self._one, self.next)
        return value

    def __exit__(self, *exc):
        # Join before checkpoints, next permutation, resume, or exception handling.
        self.pool.shutdown(wait=True, cancel_futures=True)


OLD = '''            index = permutation[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = negative_sampler(
                users_np, positives_np, data['n_items'], positive_keys, rng, k=1)'''
NEW = '''            index, negatives_np = batches.take(batch)'''


def optimized_train(module):
    original = getattr(module, '_before_gcn51_prefetch', module.train)
    source = inspect.getsource(original)
    if source.count(OLD) != 1:
        raise RuntimeError('Unexpected training source; do not patch')
    source = source.replace(OLD, NEW)
    begin = source.index('        for batch in range(n_batches):')
    end = source.index('        # Convert once', begin)
    body = source[begin:end]
    scope = dict(module.__dict__, _GCNBatchPrefetch=Batches)
    header = ("        with _GCNBatchPrefetch(permutation, cfg.batch_size, tr_u, tr_i,\n"
              "                data['n_items'], positive_keys, rng, negative_sampler) as batches:\n")
    source = source[:begin] + header + ''.join('    ' + line if line.strip() else line for line in body.splitlines(True)) + source[end:]
    exec(compile(source, '<gcn51-prefetch>', 'exec'), scope)
    return scope['train'], hashlib.sha256(inspect.getsource(original).encode()).hexdigest()


def install(screen):
    raise RuntimeError('L4 probe showed no speed gain and failed exact-state equivalence; production use disabled')
