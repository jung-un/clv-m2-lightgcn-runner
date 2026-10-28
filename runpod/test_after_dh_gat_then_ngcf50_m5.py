import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import after_dh_gat_then_ngcf50_m5 as q


class HandoffChecks(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.prior=self.root/'prior';self.run=self.prior/q.RUN_ID
        (self.run/'arms').mkdir(parents=True);(self.run/'progress/stages').mkdir(parents=True)
        (self.prior/'runpod_partition.json').write_text(json.dumps(dict(ada_seeds=list(q.SEEDS),l4_seeds=[43,44,45])))
        self.protocol=dict(source_revision=q.GAT_PIN,config={'epochs':300},code_version='stage',
                           input_hash='input',validation=False,holdout=False,early_stopping=False)
        (self.run/'protocol.json').write_text(json.dumps(self.protocol))
        self.log=self.root/'prior.log';self.log.write_text('요청한 작업 완료.')
        for seed in q.SEEDS:
            for model in q.platform.ORDER:
                mid=f'gat_{model.lower()}_lastweek'
                ident=dict(stage='stage',model_id=mid,seed=seed,config_hash=q.RUN_ID,
                           source_revision=q.GAT_PIN,input_hash='input')
                row=dict(identity=ident,model=model,seed=seed,epochs=300,
                         metrics={str(i):1. for i in range(142)},training_history=[{'epoch':i} for i in range(1,301)])
                (self.run/'arms'/f'{mid}_s{seed}.json').write_text(json.dumps(row))
                (self.run/'progress/stages'/f'stage_{mid}_s{seed}.completed.json').write_text(
                    json.dumps(dict(identity=ident,status='completed')))

    def test_all_40_durable_results_required(self):
        q.completion_check(self.prior,self.log)
        p=self.run/'arms/gat_m5_d_lastweek_s51.json';row=json.loads(p.read_text())
        row['epochs']=299;p.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError):q.completion_check(self.prior,self.log)

    def test_wrong_assignment_identity_nan_marker_and_end_log_block(self):
        assignment=self.prior/'runpod_partition.json';saved=assignment.read_text()
        assignment.write_text(json.dumps(dict(ada_seeds=[46,47,48],l4_seeds=[43,44,45])))
        with self.assertRaises(RuntimeError):q.completion_check(self.prior,self.log)
        assignment.write_text(saved)
        p=self.run/'arms/gat_m1_lastweek_s46.json';row=json.loads(p.read_text());saved=p.read_text()
        for key,value in [('seed',49),('metrics',{'0':float('nan')})]:
            p.write_text(json.dumps({**row,key:value}))
            with self.assertRaises(RuntimeError):q.completion_check(self.prior,self.log)
        p.write_text(saved);self.log.write_text('Traceback failed')
        with self.assertRaises(RuntimeError):q.completion_check(self.prior,self.log)
        self.log.write_text('요청한 작업 완료.')
        next((self.run/'progress/stages').glob('*.json')).unlink()
        with self.assertRaises(FileNotFoundError):q.completion_check(self.prior,self.log)

    def test_waits_for_exact_parent_before_m5_child(self):
        events=[]
        class Poll:
            def register(self,*a):events.append('wait')
            def poll(self,*a):return [(1,1)]
        with patch.object(q,'ROOT',self.root/'next'),patch.object(q.core,'SHARED',self.root/'shared'), \
             patch.object(q.core,'preflight'),patch.object(q,'process_identity',return_value='ticks'), \
             patch.object(q.os,'pidfd_open',return_value=1,create=True),patch.object(q.os,'close'), \
             patch.object(q.select,'poll',return_value=Poll()), \
             patch.object(q,'completion_check',side_effect=lambda:events.append('validated40')), \
             patch.object(q.subprocess,'check_output',return_value=''), \
             patch.object(q.subprocess,'Popen',side_effect=lambda *a,**k:events.append('start') or SimpleNamespace(pid=1,wait=lambda:0)):
            q.queue('wait',8880)
        self.assertLess(events.index('wait'),events.index('validated40'))
        self.assertLess(events.index('validated40'),events.index('start'))
        self.assertEqual(q.MODELS,('M5_A','M5_B','M5_C','M5_D'))

    def test_check_mode_never_starts_or_waits(self):
        with patch.object(q.core,'SHARED',self.root/'shared'),patch.object(q.core,'preflight'), \
             patch.object(q,'process_identity',return_value='ticks'),patch.object(q.subprocess,'Popen') as start:
            q.queue('check',8880)
            start.assert_not_called()

    def test_invalid_partition_rejected_before_training(self):
        with self.assertRaises(RuntimeError):q.core.run_seed50(models=('M1','M5_A'),pod=q.POD)
        with self.assertRaises(RuntimeError):q.core.run_seed50(models=q.MODELS,pod='wrong')


if __name__=='__main__':unittest.main()
