import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import clv_ngcf_storage_fast as fast
import clv_ngcf_school_storage as storage

# Reuse the storage test's in-memory Drive, without importing GPU dependencies.
tree = ast.parse(Path(__file__).with_name('test_clv_ngcf_school_storage.py').read_text())
scope = dict(copy=copy, Path=Path, hashlib=hashlib, storage=storage)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)
                             and n.name in ('Request', 'API')], type_ignores=[]), '<fake-drive>', 'exec'), scope)


class Tests(unittest.TestCase):
    def setUp(self):
        self.module = types.SimpleNamespace(**{k: getattr(storage, k) for k in
            ('START_FREE', 'MIN_FREE', 'FIELDS', 'rows_for', 'execute')})
        self.calls = 0
        def check(api, required=0, initial=False):
            self.calls += 1
            return 10_000_000_000
        self.module.check_space = check
        self.module.wait_for_space = lambda api, required: self.module.check_space(api, required)
        self.api = scope['API']()
        self.ident = types.SimpleNamespace(config_hash='test', model_id='M2', seed=50)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'checkpoint.pt'
        upload = types.ModuleType('googleapiclient.http')
        upload.MediaFileUpload = lambda path, **kw: types.SimpleNamespace(path=path)
        p = patch.dict(sys.modules, {'googleapiclient.http': upload})
        p.start(); self.addCleanup(p.stop)
        fast.install(self.module)

    def test_rotate_and_quota_requests(self):
        for epoch in range(1, 6):
            self.path.write_bytes(str(epoch).encode())
            self.module.publish(self.api, 'folder', self.path, self.ident, epoch)
        self.assertEqual(self.calls, 1)
        self.assertEqual(sorted(int(r['appProperties']['clv_epoch']) for r in self.api.rows.values()), [4, 5])
        self.assertEqual(len(self.api.deleted), 3)

    def test_conflict_preserves_files_and_invalidates_cache(self):
        self.path.write_bytes(b'a')
        self.module.publish(self.api, 'folder', self.path, self.ident, 1)
        self.path.write_bytes(b'b')
        with self.assertRaises(RuntimeError):
            self.module.publish(self.api, 'folder', self.path, self.ident, 1)
        self.assertFalse(self.api.deleted)
        self.module.check_space(self.api)
        self.assertEqual(self.calls, 2)

    def test_expiry_and_conservative_accounting(self):
        with patch.object(fast.time, 'monotonic', return_value=1):
            self.module.check_space(self.api, 2_000_000_000)
            self.assertEqual(self.module.check_space(self.api), 8_000_000_000)
        with patch.object(fast.time, 'monotonic', return_value=122):
            self.module.check_space(self.api)
        self.assertEqual(self.calls, 2)


if __name__ == '__main__':
    unittest.main()
