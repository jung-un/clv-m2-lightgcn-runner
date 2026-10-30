import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
import clv_dh_gcn_parallel as p

class Partition(unittest.TestCase):
    def test_disjoint_and_handshake(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'protocol.json').write_text('{}')
            task=NS(SEEDS=(51,42,43,44,45,46,47,48),sha=lambda _: 'fixed')
            ctx={'out':root,'original':NS(base=NS(io=NS(_atomic_json=lambda path,v:path.write_text(json.dumps(v)))))}
            with self.assertRaises(RuntimeError):p.install(task,ctx,'B')
            p.install(task,ctx,'A');p.check(ctx);p.install(task,ctx,'B');p.check(ctx)
            self.assertFalse(set(p.ASSIGNMENTS['A']) & set(p.ASSIGNMENTS['B']))
            (root/'parallel_assignment.json').write_text('{}')
            with self.assertRaises(RuntimeError):p.check(ctx)
    def test_only_assigned_arms(self):
        calls=[];old=p.report
        try:
            p.report=lambda t,c:({'completed':0},[])
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);record={'fixed':True};(root/'parallel_assignment.json').write_text(json.dumps(record))
                ctx={'out':root,'parallel_assignment':record,'parallel_role':'B'}
                task=NS(ORDER=('M1','M2'),run_new=lambda c,s,m:calls.append((s,m)))
                p.run(task,ctx)
                self.assertEqual(calls,[(s,m) for s in (46,47,48) for m in ('M1','M2')])
        finally:p.report=old
if __name__=='__main__':unittest.main()
