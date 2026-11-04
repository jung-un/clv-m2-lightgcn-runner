"""Execution changes must not change experiment identity or partition ownership."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

class NotebookTests(unittest.TestCase):
    def test_pinned_source_and_ten_epoch_storage(self):
        for part, labels, run in [('a',('M1','M2','M3'),'23565920167e'),
                                  ('b',('M4','M5_A'),'73149b37a2fb'),
                                  ('c',('M5_B',),'e7c953e6d02c')]:
            n=json.loads(Path(f'clv_hm_gcn_seed51_part_{part}_personal_colab.ipynb').read_text())
            setup=''.join(n['cells'][1]['source']);prepare=''.join(n['cells'][3]['source'])
            values={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(setup).body
                    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id in ('SOURCE_COMMIT','LABELS','_storage_fix_source')}
            self.assertEqual(values['SOURCE_COMMIT'],'2d8ac73e30e04f779e0d220846610717b24be94e')
            self.assertEqual(values['LABELS'],labels)
            self.assertEqual(values['_storage_fix_source'],Path('clv_personal_gcn51_checkpoint.py').read_text())
            self.assertIn("protocol_fix.install_gcn(screen, '"+run+"')",prepare)
            calls=[x for x in ast.walk(ast.parse(prepare)) if isinstance(x,ast.Call) and ast.unparse(x.func)=='storage.install']
            self.assertEqual(len(calls),2)
            for call in calls:self.assertEqual({x.arg:ast.literal_eval(x.value) for x in call.keywords}['drive_interval'],10)
            for cell in n['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))

if __name__=='__main__':unittest.main()
