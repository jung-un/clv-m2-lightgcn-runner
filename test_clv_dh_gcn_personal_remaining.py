"""Focused coordinator tests; no training, GPU or remote Drive calls."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import clv_dh_gcn_personal_remaining as task


class CoordinatorTests(unittest.TestCase):
    def test_plan_no_duplicate_or_reused_seed(self):
        self.assertEqual(task.SEEDS[0],51)
        self.assertEqual(set(task.SEEDS),set(range(42,52))-{49,50})
        self.assertEqual(len(set(task.SEEDS)),8)

    def test_verified_extract_and_conflict(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with ZipFile(root/'x.zip','w') as z:
                z.writestr('x.txt',b'original')
                z.writestr('manifest.json',json.dumps({'x.txt':hashlib.sha256(b'original').hexdigest()}))
            task.unpack_verified(root/'x.zip',root/'out')
            task.unpack_verified(root/'x.zip',root/'out')
            (root/'out/x.txt').write_text('changed')
            with self.assertRaises(RuntimeError): task.unpack_verified(root/'x.zip',root/'out')
            self.assertEqual((root/'out/x.txt').read_text(),'changed')

    def test_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with ZipFile(p,'w') as z:z.writestr('manifest.json','{"../bad":"x"}')
            with self.assertRaises(ValueError):task.unpack_verified(p,Path(d)/'out')

    def test_dispatch_resumes_51_only_with_original(self):
        calls=[]
        ctx={'original':types.SimpleNamespace(run_arm=lambda p,m:calls.append((51,m))), 'shared':{}}
        with patch.object(task,'run_new',side_effect=lambda c,s,m:calls.append((s,m))),patch.object(task,'report',return_value=({'completed':80},[])):
            task.run(ctx)
        self.assertEqual(len(calls),64)
        self.assertEqual(calls[:8],[(51,m) for m in task.ORDER])
        self.assertFalse(any(s in (49,50) for s,m in calls))

    def test_no_final_average_before_80(self):
        import pandas as pd
        rows=[{'seed':s,'model':m,'metrics':{'metric':float(s)}} for s in task.ALL_SEEDS for m in task.ORDER]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            io=types.SimpleNamespace(_atomic_json=lambda p,x:p.write_text(json.dumps(x)),
                                    _atomic_csv=lambda p,x:x.to_csv(p,index=False))
            o=types.SimpleNamespace(base=types.SimpleNamespace(io=io),completed=lambda p,m:None)
            ctx={'out':root,'previous':rows[:-1],'original':o,'shared':{'references':{'metrics':['metric']}}}
            status,_=task.report(ctx)
            self.assertFalse(status['complete'])
            self.assertFalse((root/'mean_std_10seeds.csv').exists())
            ctx['previous']=rows
            status,_=task.report(ctx)
            self.assertTrue(status['complete'])
            result=pd.read_csv(root/'mean_std_10seeds.csv')
            self.assertEqual(result.iloc[0]['metric__mean'],46.5)
            self.assertAlmostEqual(result.iloc[0]['metric__std'],pd.Series(range(42,52)).std())


if __name__=='__main__':unittest.main()
