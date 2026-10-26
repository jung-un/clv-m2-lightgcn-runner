"""Focused platform-adapter checks; no full dataset or final test evaluation."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('runpod_launch', Path(__file__).with_name('launch.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class PlatformChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_seed_scope(self):
        self.assertEqual(s.selected_seeds('dh-ngcf', [51, 42]), (42, 51))
        for seeds in ([], [49], [42, 42], [41], [52]):
            with self.assertRaises(ValueError): s.selected_seeds('dh-ngcf', seeds)
        self.assertEqual(s.selected_seeds('hm-ngcf', None), (49,))
        self.assertEqual(s.selected_seeds('hm-lightgcn', None), (51,))
        with self.assertRaises(ValueError): s.selected_seeds('hm-lightgcn', [50])

    def bundle(self, name, content=b'original'):
        import hashlib
        p = self.root/'input.zip'
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr('manifest.json', json.dumps({name: hashlib.sha256(content).hexdigest()}))
            z.writestr(name, content)
        return p

    def test_verified_unpack_and_no_overwrite(self):
        bundle = self.bundle('raw/x')
        target = self.root/'data'
        s.unpack(bundle, target)
        s.unpack(bundle, target)
        self.assertEqual((target/'raw/x').read_bytes(), b'original')
        (target/'raw/x').write_bytes(b'keep this')
        with self.assertRaises(RuntimeError): s.unpack(bundle, target)
        self.assertEqual((target/'raw/x').read_bytes(), b'keep this')

    def test_zip_path_traversal_rejected(self):
        with self.assertRaises(RuntimeError): s.unpack(self.bundle('../outside'), self.root/'data')
        self.assertFalse((self.root/'outside').exists())

    def test_wrong_raw_and_missing_files_rejected(self):
        with self.assertRaises(RuntimeError): s.verify_files('hm-ngcf', self.root)
        p = self.root/'data/raw/hm/transactions_train.parquet'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'wrong converted parquet')
        with self.assertRaisesRegex(RuntimeError, '다른 입력'): s.verify_files('hm-ngcf', self.root)

    def test_uploaded_hm_raw_is_verified_before_copy(self):
        import hashlib
        source = self.root/'upload.parquet'
        source.write_bytes(b'verified test bytes')
        name = 'raw/hm/transactions_train.parquet'
        with patch.dict(s.HM_FILES, {name: hashlib.sha256(source.read_bytes()).hexdigest()}):
            s.install_hm_raw(source, self.root)
            s.install_hm_raw(source, self.root)
            self.assertEqual((self.root/'data'/name).read_bytes(), source.read_bytes())
            source.write_bytes(b'wrong bytes')
            with self.assertRaises(RuntimeError): s.install_hm_raw(source, self.root)
            self.assertEqual((self.root/'data'/name).read_bytes(), b'verified test bytes')

    def test_gpu_absence_stops_before_research_model(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaises(RuntimeError): s.cuda_check(torch)

    def test_concurrent_process_lock(self):
        with s.exclusive(self.root/'job.lock'):
            with self.assertRaises(RuntimeError):
                with s.exclusive(self.root/'job.lock'): pass
        with s.exclusive(self.root/'job.lock'): pass

    def test_environment_changes_fail_closed(self):
        p = self.root/'env.json'
        s.bind_environment(p, {'gpu': 'L4'})
        s.bind_environment(p, {'gpu': 'L4'})
        with self.assertRaises(RuntimeError): s.bind_environment(p, {'gpu': 'A100'})
        self.assertEqual(json.loads(p.read_text()), {'gpu': 'L4'})

    def test_final_evaluation_not_retried_after_interruption(self):
        calls = []
        original = lambda *a, **k: calls.append(1) or {'metric': 1}
        screen = SimpleNamespace(base=SimpleNamespace(fixed=SimpleNamespace(_evaluate=original)))
        p = {'run_dir': self.root}
        with s.evaluation_once(screen, p, 'M1', 42):
            self.assertEqual(screen.base.fixed._evaluate(), {'metric': 1})
        with s.evaluation_once(screen, p, 'M1', 42):
            with self.assertRaises(RuntimeError): screen.base.fixed._evaluate()
        self.assertEqual(len(calls), 1)
        self.assertIs(screen.base.fixed._evaluate, original)

    def test_storage_failure_preserves_latest_and_previous(self):
        class Store:
            def __init__(inner): inner.latest_checkpoint = self.root/'latest.pt'
            def save_epoch(inner, *args, **kwargs): raise OSError('disk write failed')
        screen = SimpleNamespace(SafeProgressStore=Store, STORAGE_GUARD=None)
        s.install_storage(screen, 'hm-ngcf', self.root)
        store = screen.SafeProgressStore()
        store.latest_checkpoint.write_bytes(b'old state')
        with patch.object(s, 'guard_space'):
            with self.assertRaises(OSError):
                store.save_epoch(SimpleNamespace(parameters=lambda: []), None, None)
        self.assertEqual(store.latest_checkpoint.read_bytes(), b'old state')
        self.assertEqual(store.latest_checkpoint.with_suffix('.previous.pt').read_bytes(), b'old state')

    def test_low_storage_blocks(self):
        with patch.object(s.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaises(RuntimeError): s.guard_space(self.root)


if __name__ == '__main__':
    unittest.main()
