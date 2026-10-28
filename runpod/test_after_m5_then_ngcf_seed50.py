"""Small fail-closed queue checks; no research data or GPU training."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import after_m5_then_ngcf_seed50 as q


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'results/hm-ngcf'/q.PRIOR_RUN
        (self.run/'arms').mkdir(parents=True)
        (self.run/'progress/stages').mkdir(parents=True)
        self.protocol = dict(source_revision=q.storage.PIN, config={'epochs':300},
                             code_version='stage49', input_hash='input')
        (self.run/'protocol.json').write_text(json.dumps(self.protocol))
        (self.run.parent/'runpod_m5_partition.json').write_text(json.dumps(
            {'seeds':[49], 'models':list(q.PRIOR_MODELS)}))
        (self.root/'logs').mkdir()
        (self.root/'logs/current.log').write_text('M5_A~D 요청 완료.')
        for label in q.PRIOR_MODELS:
            mid = f'ngcf_{label.lower()}_lastweek'
            ident = dict(stage='stage49', model_id=mid, seed=49,
                         config_hash=q.PRIOR_RUN, source_revision=q.storage.PIN, input_hash='input')
            row = dict(identity=ident,model=label,seed=49,epochs=300,
                       training_history=[{'epoch':i} for i in range(1,301)],
                       metrics={str(i):1. for i in range(142)})
            (self.run/'arms'/f'{mid}_s49.json').write_text(json.dumps(row))
            (self.run/'progress/stages'/f'stage49_{mid}_s49.completed.json').write_text(
                json.dumps({'identity':ident,'status':'completed'}))

    def test_all_four_m5_durable_results_required(self):
        q.completion_check(self.root)
        path=self.run/'arms/ngcf_m5_d_lastweek_s49.json'
        row=json.loads(path.read_text());row['epochs']=299;path.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.root)

    def test_missing_marker_and_error_end_block(self):
        (self.root/'logs/current.log').write_text('Traceback failed')
        with self.assertRaises(RuntimeError): q.completion_check(self.root)
        (self.root/'logs/current.log').write_text('M5_A~D 요청 완료.')
        next((self.run/'progress/stages').glob('*.json')).unlink()
        with self.assertRaises(FileNotFoundError): q.completion_check(self.root)

    def test_nonfinite_or_wrong_seed_block(self):
        path=self.run/'arms/ngcf_m5_d_lastweek_s49.json'
        row=json.loads(path.read_text());row['metrics']['0']=float('nan');path.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.root)
        row['metrics']['0']=1.;row['identity']['seed']=50;path.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.root)

    def test_science_permits_seed_and_driver_only(self):
        baseline={k:False for k in q.storage.SCIENCE_KEYS}
        baseline.update(config={'seeds':[49],'epochs':300,'lr':.0005},
                        code_sha256={'clv_hm_ngcf_final8_seed49.py':'old','model.py':'fixed'})
        current=json.loads(json.dumps(baseline));current['config']['seeds']=(50,)
        current['source_revision']=q.PIN
        current['code_sha256']['clv_hm_ngcf_final8_seed49.py']='new'
        q.check_science(current,baseline)
        current['config']['lr']=.01
        with self.assertRaises(RuntimeError):q.check_science(current,baseline)
        current['config']['lr']=.0005;current['code_sha256']['model.py']='different'
        with self.assertRaises(RuntimeError):q.check_science(current,baseline)

    def test_queue_does_not_start_before_parent_and_completion_verification(self):
        events=[]
        class Poll:
            def register(self,*a):events.append('wait_registered')
            def poll(self,*a):return [(1,1)]
        child=SimpleNamespace(pid=123,wait=lambda:0)
        with patch.object(q,'SHARED',self.root/'shared'),patch.object(q,'ROOT',self.root/'next'), \
             patch.object(q,'preflight'),patch.object(q,'process_identity',return_value='ticks'), \
             patch.object(q.os,'pidfd_open',return_value=1,create=True),patch.object(q.os,'close'), \
             patch.object(q.select,'poll',return_value=Poll()), \
             patch.object(q,'completion_check',side_effect=lambda:events.append('complete_verified')), \
             patch.object(q.subprocess,'check_output',return_value=''), \
             patch.object(q.subprocess,'Popen',side_effect=lambda *a,**k:events.append('start') or child) as start:
            q.queue('wait',9398)
        self.assertLess(events.index('wait_registered'),events.index('complete_verified'))
        self.assertLess(events.index('complete_verified'),events.index('start'))
        self.assertEqual(start.call_args.args[0][-1],'run')
        self.assertEqual(q.MODELS,('M1','M2','M3','M4'))

    def test_failed_parent_and_busy_gpu_never_start_child(self):
        for failure in ('incomplete','gpu_busy'):
            with patch.object(q,'SHARED',self.root/'shared'),patch.object(q,'ROOT',self.root/'next'), \
                 patch.object(q,'preflight'),patch.object(q,'process_identity',return_value=None), \
                 patch.object(q,'completion_check',side_effect=RuntimeError('incomplete') if failure=='incomplete' else None), \
                 patch.object(q.subprocess,'check_output',return_value='999'), \
                 patch.object(q.subprocess,'Popen') as start:
                with self.assertRaises(RuntimeError):q.queue('wait',9398)
                start.assert_not_called()


if __name__=='__main__': unittest.main()
