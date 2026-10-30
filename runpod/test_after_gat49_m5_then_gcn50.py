import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import after_gat49_m5_then_gcn50 as q


class CompletionTests(unittest.TestCase):
    def test_requires_all_four_completed_original_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/'results/hm-gat'
            run = output/'hash123'
            def save(p, value):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value))
            save(root/'queue_status.json', {'phase':'completed_gat_m5'})
            save(output/'runpod_m5_partition.json', {'seeds':[49], 'models':list(q.MODELS[4:])})
            p = dict(config={'seeds':[49], 'epochs':300}, code_version='v1', source_revision='pin', input_hash='input')
            save(run/'protocol.json', p)
            for label in q.MODELS[4:]:
                mid = 'gat_'+label.lower()+'_lastweek'
                ident = dict(stage='v1', model_id=mid, seed=49, config_hash='hash123', source_revision='pin', input_hash='input')
                row = dict(identity=ident, model=label, seed=49, epochs=300,
                           training_history=[{'epoch':i} for i in range(1,301)],
                           metrics={str(i):.01 for i in range(142)})
                save(run/'arms'/f'{mid}_s49.json', row)
                save(run/'progress/stages'/f'v1_{mid}_s49.completed.json', dict(identity=ident,status='completed'))
            self.assertEqual(q.completion_check(root), p)
            save(root/'queue_status.json', {'phase':'failed_no_automatic_retry'})
            with self.assertRaises(RuntimeError): q.completion_check(root)
            save(root/'queue_status.json', {'phase':'completed_gat_m5'})
            row['epochs'] = 299
            save(run/'arms'/f'{mid}_s49.json', row)
            with self.assertRaises(RuntimeError): q.completion_check(root)


if __name__ == '__main__':
    unittest.main()

