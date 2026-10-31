"""Focused audit regression; extracts actual functions without importing torch."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).parent

def function(file, name, scope):
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), scope)
    return scope[name]

gate = function('lightgcn_clv_m3_centered_value_graph.py', 'check_gates', {'CenteredGraphConfig': object})
compare = function('clv_hm_ngcf_final8_seed49.py', 'compare_m3_audit',
    {'np': np, 'base': SimpleNamespace(fixed=SimpleNamespace(graph=SimpleNamespace(check_gates=gate)))})
OLD = dict(coefficient_of_variation=.20008521133516644, customer_weight_vs_degree=.006917207044767613,
 item_weight_vs_price=-.03610003501492243, item_weight_vs_popularity=-.07292047735297143,
 max_customer_mass_error=4.935481652997472e-10, weight_min=.11659132670740294, weight_max=15.92239747895888)
NEW = dict(coefficient_of_variation=.20008521133516655, customer_weight_vs_degree=.00938422327390731,
 item_weight_vs_price=-.03490933101819989, item_weight_vs_popularity=-.06927069046267695,
 max_customer_mass_error=4.935483133294838e-10, weight_min=.11659132670740278, weight_max=15.9223974789589)
CFG = SimpleNamespace(max_degree_correlation=.05, max_price_correlation=.2, max_popularity_correlation=.2)

class Audit(unittest.TestCase):
 def test_graphsage49_uses_checked_comparison(self):
  tree=ast.parse((ROOT/'clv_hm_graphsage_final8_seed49.py').read_text())
  prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
    and n.func.id=='compare_m3_audit' for n in ast.walk(prepare)))
  self.assertIn('m3_reference_comparison',ast.unparse(prepare))
 def test_graphsage_partition_uses_same_checked_comparison(self):
  tree=ast.parse((ROOT/'clv_hm_graphsage_m5_seed50.py').read_text())
  self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module=='clv_hm_ngcf_final8_seed49'
    and any(a.name=='compare_m3_audit' for a in n.names) for n in tree.body))
  prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
    and n.func.id=='compare_m3_audit' for n in ast.walk(prepare)))
  self.assertIn('m3_reference_comparison',ast.unparse(prepare))
 def test_reproduces_old_failure_and_preserves_actual(self):
  self.assertFalse(all(np.isclose(NEW[k],v,rtol=1e-8,atol=1e-10) for k,v in OLD.items()))
  result=compare(OLD,NEW,CFG)
  self.assertEqual(result['reference'],OLD); self.assertEqual(result['current'],NEW)
 def test_identical(self): self.assertEqual(compare(OLD,OLD,CFG)['differences'],{})
 def test_bad_inputs_remain_blocked(self):
  for key,value in [('weight_max',16),('coefficient_of_variation',.21),('customer_weight_vs_degree',.06),('item_weight_vs_price',.21),('item_weight_vs_popularity',-.21),('weight_min',float('nan'))]:
   with self.subTest(key=key),self.assertRaises(RuntimeError): compare(OLD,{**NEW,key:value},CFG)
 def test_missing_key(self):
  with self.assertRaises(RuntimeError): compare(OLD,{k:v for k,v in NEW.items() if k!='weight_min'},CFG)
 def test_nonconverged_rank_change(self):
  with self.assertRaises(RuntimeError): compare({**OLD,'max_customer_mass_error':1e-7},{**NEW,'max_customer_mass_error':1e-7},CFG)

if __name__=='__main__': unittest.main()
