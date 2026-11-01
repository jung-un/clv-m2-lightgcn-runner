import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import clv_hm_graphsage_school_single as single


class PartitionTests(unittest.TestCase):
    def test_only_requested_arms(self):
        for label in ('M3', 'M4'):
            single.validate(label)
        for label in ('M1', 'M2', 'M5_A', ''):
            with self.assertRaises(ValueError): single.validate(label)

    def test_export_single_complete_and_partial(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            arm = root / 'm3.json'
            arm.write_text('{"model":"M3"}')
            screen = types.SimpleNamespace(completed=lambda p,l: {'metrics': {'recall@10': .1}},
                                           arm_path=lambda p,l: arm)
            p, status = single.export(screen, {'run_dir':root}, 'M3', root/'result.zip')
            self.assertTrue(status['complete'])
            self.assertEqual(status['models'], ['M3'])
            with ZipFile(p) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('arms/m3.json', z.namelist())
            screen.completed = lambda p,l: None
            p,status=single.export(screen, {'run_dir':root}, 'M4', root/'m4.zip')
            self.assertFalse(status['complete'])
            self.assertTrue(p.name.endswith('_partial.zip'))

    def test_capacity_fails_without_reserve(self):
        api=types.SimpleNamespace(about=lambda:types.SimpleNamespace(get=lambda **k:None))
        io=types.SimpleNamespace(execute=lambda r:{'user':{'emailAddress':single.ACCOUNT},
            'storageQuota':{'usage':'4740000000','limit':'10000000000'}},listing=lambda *a:[])
        with TemporaryDirectory() as d, patch.dict('sys.modules',{'clv_ngcf_school_storage':io}):
            with self.assertRaises(RuntimeError): single.capacity_check(None,api,d)
            io.execute=lambda r:{'user':{'emailAddress':single.ACCOUNT},
                'storageQuota':{'usage':'1000000000','limit':'10000000000'}}
            self.assertEqual(single.capacity_check(None,api,d),9000000000)


if __name__=='__main__': unittest.main()
