"""Partition and storage failure checks without research data/test evaluation."""
from contextlib import nullcontext
import errno
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hm_ngcf_m5 as s


class PartitionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.shared = Path(self.temp.name) / 'clv'
        self.root = Path(self.temp.name) / 'm5'
        self.output = self.shared / 'results/hm-ngcf'
        self.output.mkdir(parents=True)
        self.assignment = self.output / 'runpod_invocation_20261026.json'
        self.assignment.write_text(json.dumps({'seeds': [49], 'models': ['M1','M2','M3','M4']}))
        self.protocol = {key: False for key in s.SCIENCE_KEYS}
        self.protocol.update(source_revision=s.PIN, config={'epochs': 300, 'out_dir': 'old'})
        protocol_path = self.output / 'baseline/protocol.json'
        protocol_path.parent.mkdir()
        protocol_path.write_text(json.dumps(self.protocol))
        (self.shared / 'data').mkdir()
        (self.shared / 'sources' / s.PIN).mkdir(parents=True)

    def test_only_m5_and_fixed_source(self):
        self.assertEqual(s.MODELS, ('M5_A','M5_B','M5_C','M5_D'))
        self.assertEqual(s.PIN, s.platform.JOBS['hm-ngcf']['pin'])
        self.assertTrue(s.check_existing(self.shared))
        for models in (list(s.platform.ORDER), ['M1','M5_A'], []):
            self.assignment.write_text(json.dumps({'seeds': [49], 'models': models}))
            with self.assertRaises(RuntimeError): s.check_existing(self.shared)

    def test_existing_artifact_blocks_even_without_completed_result(self):
        for suffix in ('ngcf_m5_a_lastweek_s49.json', 'clv-hm_ngcf_m5_d_lastweek_s49_latest.pt', 'M5_C_s49.json'):
            artifact = self.output / suffix
            artifact.write_text('keep')
            with self.assertRaises(RuntimeError): s.check_existing(self.shared)
            self.assertEqual(artifact.read_text(), 'keep')
            artifact.unlink()

    def test_missing_record_and_wrong_pin_fail_closed(self):
        self.assignment.unlink()
        with self.assertRaises(RuntimeError): s.check_existing(self.shared)

    def test_input_links_and_separate_output(self):
        s.link_shared_inputs(self.root, self.shared)
        s.link_shared_inputs(self.root, self.shared)
        self.assertEqual((self.root / 'data').resolve(), (self.shared / 'data').resolve())
        with self.assertRaises(RuntimeError): s.link_shared_inputs(self.shared, self.shared)
        with self.assertRaises(RuntimeError): s.link_shared_inputs(self.shared / 'nested', self.shared)
        (self.root / 'data').unlink()
        (self.root / 'data').mkdir()
        with self.assertRaises(RuntimeError): s.link_shared_inputs(self.root, self.shared)

    def test_science_allows_paths_only(self):
        current = json.loads(json.dumps(self.protocol))
        current['config']['out_dir'] = 'new'
        s.check_science(current, [self.protocol])
        for field in ('epochs', 'batch_size', 'lr'):
            changed = json.loads(json.dumps(current))
            changed['config'][field] = 1
            with self.assertRaises(RuntimeError): s.check_science(changed, [self.protocol])
        current['feature_sha256'] = 'different'
        with self.assertRaises(RuntimeError): s.check_science(current, [self.protocol])

    def test_environment_mismatch_blocks(self):
        env = {key: 'fixed' for key in ('source_commit','input_sha256','python','packages','cuda','gpu',
                                      'capability','tf32_matmul','matmul_precision')}
        (self.output / 'runpod_environment.json').write_text(json.dumps(env))
        s.check_environment(env, self.shared)
        with self.assertRaises(RuntimeError): s.check_environment(dict(env, gpu='different'), self.shared)

    def test_execute_prepares_once_runs_only_m5_and_never_writes_baseline(self):
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        calls, preparations = [], []
        prepared = {'protocol': self.protocol, 'data': {'splits': {'test': {}}, 'loss_w': None,
                    'data_stats': {'min_item_inter': 1}}}
        screen = SimpleNamespace(prepare=lambda cfg: preparations.append(1) or prepared,
                                 run_arm=lambda p, label: calls.append(label))
        with patch.dict('sys.modules', {'torch': SimpleNamespace()}), \
             patch.object(s.platform, 'verify_files', return_value={}), \
             patch.object(s.platform, 'cuda_check'), \
             patch.object(s.platform, 'load_runner', return_value=(screen, None)), \
             patch.object(s.platform, 'guard_space'), \
             patch.object(s.platform, 'environment', return_value={}), \
             patch.object(s, 'check_environment'), patch.object(s, 'install_storage'), \
             patch.object(s.platform, 'evaluation_once', side_effect=lambda *a: nullcontext()), \
             patch.object(s.platform, 'export'):
            s.execute('run', self.root, self.shared)
        self.assertEqual(preparations, [1])
        self.assertEqual(tuple(calls), s.MODELS)
        self.assertEqual(before, {p.relative_to(self.output): p.read_bytes()
                                 for p in self.output.rglob('*') if p.is_file()})


class StorageChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.latest = Path(self.temp.name) / 'latest.pt'
        self.latest.write_bytes(b'epoch1 model Adam RNG')
        self.previous = self.latest.with_suffix('.previous.pt')

    def test_hardlink_keeps_old_state_after_atomic_publish_without_copy(self):
        with patch.object(s.platform, 'sha', side_effect=AssertionError('unneeded network read')):
            self.assertEqual(s.preserve_previous(self.latest), 'hardlink')
        self.assertTrue(os.path.samefile(self.latest, self.previous))
        new = self.latest.with_suffix('.new')
        new.write_bytes(b'epoch2 model Adam RNG')
        os.replace(new, self.latest)
        self.assertEqual(self.previous.read_bytes(), b'epoch1 model Adam RNG')
        self.assertEqual(self.latest.read_bytes(), b'epoch2 model Adam RNG')
        s.preserve_previous(self.latest)
        self.assertEqual(self.previous.read_bytes(), b'epoch2 model Adam RNG')

    def test_unsupported_filesystem_uses_verified_copy(self):
        with patch.object(s.os, 'link', side_effect=OSError(errno.EOPNOTSUPP, 'unsupported')):
            self.assertEqual(s.preserve_previous(self.latest), 'verified_copy')
        self.assertFalse(os.path.samefile(self.latest, self.previous))
        self.assertEqual(self.latest.read_bytes(), self.previous.read_bytes())

    def test_corrupt_copy_preserves_both_old_files(self):
        self.previous.write_bytes(b'epoch0')
        with patch.object(s.os, 'link', side_effect=OSError(errno.EOPNOTSUPP, 'unsupported')), \
             patch.object(s.platform, 'sha', return_value='corrupt'):
            with self.assertRaises(RuntimeError): s.preserve_previous(self.latest)
        self.assertEqual(self.latest.read_bytes(), b'epoch1 model Adam RNG')
        self.assertEqual(self.previous.read_bytes(), b'epoch0')

    def test_stale_temp_and_io_failures_stop_without_overwrite(self):
        temporary = self.previous.with_suffix(self.previous.suffix + '.tmp')
        temporary.write_bytes(b'preserve')
        with self.assertRaises(RuntimeError): s.preserve_previous(self.latest)
        self.assertEqual(temporary.read_bytes(), b'preserve')
        temporary.unlink()
        with patch.object(s.os, 'link', side_effect=OSError(errno.ENOSPC, 'full')):
            with self.assertRaises(OSError): s.preserve_previous(self.latest)
        self.assertEqual(self.latest.read_bytes(), b'epoch1 model Adam RNG')

    def test_save_failure_preserves_current_and_previous(self):
        latest = self.latest
        class Original:
            def __init__(self): self.latest_checkpoint = latest
            def save_epoch(self, *args, **kw): raise OSError('save failed before atomic replace')
        screen = SimpleNamespace(SafeProgressStore=Original)
        s.install_storage(screen, self.latest.parent)
        with patch.object(s.platform, 'guard_space'):
            with self.assertRaises(OSError):
                screen.SafeProgressStore().save_epoch(SimpleNamespace(parameters=lambda: []), None, None)
        self.assertEqual(self.latest.read_bytes(), self.previous.read_bytes())


if __name__ == '__main__':
    unittest.main()
