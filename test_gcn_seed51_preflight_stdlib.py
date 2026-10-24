"""Seed-only equivalence and school quota cache failure behavior; no GPU required."""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).parent

class PreflightTests(unittest.TestCase):
    def test_only_seed_changes(self):
        original = (ROOT/'clv_dh_gcn_final8_seed50.py').read_text()
        new = (ROOT/'clv_dh_gcn_final8_seed51.py').read_text()
        normalized = re.sub(r'\b51\b', '50', new.replace('seed51','seed50').replace('_s51','_s50'))
        self.assertEqual(original, normalized)
        self.assertIn('for k in (10, 20, 50)', new)

    def notebook(self):
        return json.loads((ROOT/'clv_dh_gcn_final8_seed51_school_colab.ipynb').read_text())

    def test_cells_parse(self):
        for cell in self.notebook()['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])

    def test_cached_inventory_charges_each_write_and_fails_closed(self):
        tree = ast.parse(''.join(self.notebook()['cells'][3]['source']))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=='check_space')
        calls=[]
        ns=dict(_quota_cache=None, _quota_at=0, _quota_charged=0, OTHER_RUN_RESERVE_GB=5.0,
                query_space=lambda required: (calls.append(required) or ('school', 6*10**9)))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<quota>','exec'),ns)
        ns['check_space'](400*10**6)
        ns['check_space'](400*10**6)
        with self.assertRaises(RuntimeError):ns['check_space'](400*10**6)
        self.assertEqual(len(calls),1)
        ns['_quota_at']=-10**12
        ns['check_space'](400*10**6)
        self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()
