import copy
import json
from pathlib import Path
import tempfile
import unittest
from clv_m2_protocol_roundoff import differences, guard


class RoundoffTests(unittest.TestCase):
    def setUp(self):
        self.old = {'config': {'seed': 49, 'epochs': 300}, 'input_hash': 'same',
                    'feature_sha256': 'same', 'dataset_initialization': {'rho': .1},
                    'feature_diagnostics': {'item_v_std': .06182940367908841,
                     'user_n_reliability_mean': .5399506092071533,
                     'user_v_reliability_mean': .5399506092071533, 'rows': 100}}
        self.new = copy.deepcopy(self.old)
        self.new['feature_diagnostics'].update(item_v_std=.061829403679088346,
            user_n_reliability_mean=.5399507880210876,
            user_v_reliability_mean=.5399507880210876)

    def test_actual_failure_and_preserved_bytes(self):
        self.assertNotEqual(self.old, self.new)  # Original exact comparison fails.
        self.assertEqual(len(differences(self.old, self.new)), 3)
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'protocol.json'; p.write_text(json.dumps(self.old, indent=2))
            before = p.read_bytes()
            actual = guard(p, self.new, lambda path,obj: path.write_text(json.dumps(obj)))
            self.assertEqual(actual, self.old); self.assertEqual(p.read_bytes(), before)
            self.assertEqual(len(list(Path(root).glob('diagnostic_roundoff_*.json'))), 1)

    def test_real_changes_rejected(self):
        for key,value in [('input_hash','changed'), ('feature_sha256','changed'),
                          ('config',{'seed':51,'epochs':300}), ('dataset_initialization',{'rho':.11})]:
            x=copy.deepcopy(self.new); x[key]=value
            with self.assertRaises(RuntimeError): differences(self.old,x)
        for key,value in [('rows',101), ('user_n_reliability_mean', .54),
                          ('item_v_std',float('nan'))]:
            x=copy.deepcopy(self.new); x['feature_diagnostics'][key]=value
            with self.assertRaises(RuntimeError): differences(self.old,x)


if __name__ == '__main__': unittest.main()
