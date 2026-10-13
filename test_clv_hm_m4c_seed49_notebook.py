"""CPU-only checks of the M4-C notebook orchestration, not model training."""
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


NOTEBOOK = Path(__file__).with_name('clv_hm_m4c_lastweek_seed49_colab.ipynb')
CELLS = json.loads(NOTEBOOK.read_text())['cells']
COMMIT = 'c62e861395c128a7207322b70a7e53f9364e8caf'


@dataclass
class Config:
    out_dir: str
    seeds: tuple = tuple(range(42, 52))
    epochs: int = 300
    input_days: int = 365


class NotebookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(self.tmp.name)
        self.run = Path(self.tmp.name) / 'same-run'
        self.run.mkdir()
        protocol = {'code_version': 'original-version', 'config': {
            'out_dir': self.tmp.name, 'seeds': list(range(42, 52)),
            'epochs': 300, 'input_days': 365},
            'source_revision': COMMIT, 'input_hash': 'input-hash'}
        (self.run / 'protocol.json').write_text(json.dumps(protocol))
        self.rows = {}
        self.screen = SimpleNamespace(
            CODE_VERSION='original-version', M1='m1', M3='m3', M4_C='m4c', M5_C='m5c',
            ACCURACY=('recall@10',), ECONOMIC=('vndcg@10',),
            configure=lambda: self.cfg,
            _load_completed_arm=lambda context, model, seed: self.rows.get(model),
            prepare=Mock(), run_arm=Mock(),
        )
        self.env = {'screen': self.screen, 'json': json, 'Path': Path, 'SOURCE_COMMIT': COMMIT,
                    'torch': SimpleNamespace(cuda=SimpleNamespace(
                        is_available=lambda: True, get_device_name=lambda _: 'NVIDIA L4'))}

    def execute(self, index):
        with redirect_stdout(io.StringIO()):
            exec(compile(''.join(CELLS[index]['source']), str(NOTEBOOK), 'exec'), self.env)

    def test_notebook_schema_and_all_code_cells_compile(self):
        notebook = json.loads(NOTEBOOK.read_text())
        self.assertEqual(notebook['nbformat'], 4)
        self.assertEqual(notebook['metadata']['colab']['gpuType'], 'L4')
        calls = []
        for cell in CELLS:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                code = ''.join(cell['source'])
                compile(code, str(NOTEBOOK), 'exec')
                calls.extend(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)
                             and n.func.attr in ('run_arm', 'run_batch', 'run_final'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), 'screen.run_arm(prepared, cfg, screen.M4_C, SEED)')

    def test_completed_m4_skips_preparation_training_and_evaluation(self):
        self.rows['m4c'] = {'model_id': 'm4c', 'seed': 49,
                           'metrics': {'recall@10': .1, 'vndcg@10': .2}}
        self.execute(2)
        self.execute(3)
        self.screen.prepare.assert_not_called()
        self.screen.run_arm.assert_not_called()

    def test_only_m4_is_trained_even_if_all_references_missing(self):
        self.execute(2)
        prepared = dict(self.env['saved'], data={'splits': {'test': None}})
        # Real prepare keeps tuples; saved JSON has lists.
        prepared['protocol'] = dict(prepared['protocol'])
        prepared['protocol']['config'] = dict(prepared['protocol']['config'], seeds=tuple(range(42, 52)))
        self.screen.prepare.return_value = prepared
        self.execute(3)
        self.screen.run_arm.assert_called_once_with(prepared, self.cfg, 'm4c', 49)

    def test_mismatched_input_stops_before_training(self):
        self.execute(2)
        self.screen.prepare.return_value = dict(self.env['saved'], input_hash='changed')
        with self.assertRaisesRegex(AssertionError, 'input_hash'):
            self.execute(3)
        self.screen.run_arm.assert_not_called()

    def test_missing_existing_run_fails_closed(self):
        path = self.run / 'protocol.json'
        data = json.loads(path.read_text())
        data['source_revision'] = 'other-revision'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, '0개'):
            self.execute(2)
        self.screen.prepare.assert_not_called()

    def test_duplicate_matching_runs_fail_closed(self):
        other = Path(self.tmp.name) / 'another'
        other.mkdir()
        (other / 'protocol.json').write_text((self.run / 'protocol.json').read_text())
        with self.assertRaisesRegex(RuntimeError, '2개'):
            self.execute(2)

    def test_invalid_saved_metrics_fail_closed(self):
        self.rows['m1'] = {'model_id': 'm1', 'seed': 49,
                          'metrics': {'recall@10': float('nan'), 'vndcg@10': .2}}
        with self.assertRaisesRegex(RuntimeError, '지표'):
            self.execute(2)

    def test_report_without_references_then_refresh_uses_only_saved_metrics(self):
        import pandas as pd
        self.rows['m4c'] = {'model_id': 'm4c', 'seed': 49,
                           'metrics': {'recall@10': .12, 'vndcg@10': .2}, 'diagnostics': {}}
        self.execute(2)
        self.screen._tables = lambda rows: (
            pd.DataFrame([{'model_id': r['model_id'], 'seed': r['seed'], **r['metrics']} for r in rows]),
            pd.DataFrame([{'model_id': r['model_id'], 'seed': r['seed'], **r['diagnostics']} for r in rows]))
        def csv(path, table):
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
        self.screen.io = SimpleNamespace(
            _atomic_csv=csv,
            _atomic_json=lambda path, value: path.write_text(json.dumps(value)))
        self.screen._arm_path = lambda context, model, seed: context['run_dir'] / f'{model}_{seed}.json'
        with patch.dict('sys.modules', {'IPython.display': SimpleNamespace(display=lambda _: None)}):
            self.execute(4)
            self.assertEqual(self.env['status']['missing_references'], ['m1', 'm3', 'm5c'])
            self.assertTrue(self.env['comparison'].empty)
            self.rows['m1'] = {'model_id': 'm1', 'seed': 49,
                              'metrics': {'recall@10': .1, 'vndcg@10': .2}, 'diagnostics': {}}
            self.execute(4)
        self.assertEqual(len(self.env['comparison']), 2)
        delta = self.env['comparison'].query("metric == 'recall@10'").iloc[0]
        self.assertAlmostEqual(delta['relative_change_pct'], 20.)
        self.assertFalse(self.env['status']['final_decision_available'])
        self.assertTrue(all(Path(p).is_file() for p in self.env['paths'].values()))
        self.assertFalse((self.run / 'partial_reports').exists())
        self.screen.prepare.assert_not_called()
        self.screen.run_arm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
