"""Guarded one-shot handoff; no research training or evaluation."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import after_ngcf_then_gat as q
import hm_gat_m5 as g
import hm_ngcf_m5 as storage


class HandoffChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.shared = Path(self.temp.name) / 'clv'
        self.root = Path(self.temp.name) / 'gat_m5'
        self.run = self.shared / 'results/hm-ngcf' / q.RUN_ID
        (self.run / 'arms').mkdir(parents=True)
        (self.run / 'progress/stages').mkdir(parents=True)
        self.protocol = dict(source_revision=q.NGCF_PIN, config={'epochs':300},
                             code_version='clv-hm-ngcf-final8-seed49-v1', input_hash='fixed')
        (self.run / 'protocol.json').write_text(json.dumps(self.protocol))
        (self.run.parent / 'runpod_invocation_1.json').write_text(json.dumps(
            {'seeds':[49], 'models':list(q.MODELS)}))
        for label in q.MODELS:
            mid=f'ngcf_{label.lower()}_lastweek'
            identity=dict(stage=self.protocol['code_version'],model_id=mid,seed=49,
                          config_hash=q.RUN_ID,source_revision=q.NGCF_PIN,input_hash='fixed')
            row=dict(identity=identity,model=label,seed=49,epochs=300,
                     training_history=[{'epoch':i} for i in range(1,301)],
                     metrics={f'metric_{i}':1.0 for i in range(142)})
            (self.run/'arms'/f'{mid}_s49.json').write_text(json.dumps(row))
            (self.run/'progress/stages'/f"{identity['stage']}_{mid}_s49.completed.json").write_text(
                json.dumps({'identity':identity,'status':'completed'}))
        (self.shared.parent / 'hm_resume.log').write_text('요청한 작업 완료. Pod 요금은 계속 발생')

    def test_all_four_durable_complete_results_and_end_log_required(self):
        q.completion_check(self.shared)
        file=self.run/'arms/ngcf_m4_lastweek_s49.json'
        row=json.loads(file.read_text());row['epochs']=299;file.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)

    def test_training_history_and_identity_changes_block(self):
        file=self.run/'arms/ngcf_m3_lastweek_s49.json'
        row=json.loads(file.read_text());row['training_history'].pop();file.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)
        row['training_history']=[{'epoch':i} for i in range(1,301)]
        row['identity']['seed']=50;file.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)

    def test_missing_complete_marker_and_error_exit_block(self):
        marker=next((self.run/'progress/stages').glob('*m4*.json'))
        saved=marker.read_text();marker.unlink()
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)
        marker.write_text(saved)
        (self.shared.parent/'hm_resume.log').write_text('Traceback\nfailed')
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)

    def test_nonfinite_metrics_and_reassigned_models_block(self):
        file=self.run/'arms/ngcf_m4_lastweek_s49.json'
        saved=file.read_text();row=json.loads(saved);row['metrics']['metric_0']=float('nan')
        file.write_text(json.dumps(row))
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)
        file.write_text(saved)
        (self.run.parent/'runpod_invocation_2.json').write_text(json.dumps({'seeds':[49],'models':['M5_A']}))
        with self.assertRaises(RuntimeError): q.completion_check(self.shared)

    def test_waits_for_exact_process_before_completion_checks_and_child_start(self):
        events=[]
        child=SimpleNamespace(pid=9000, wait=lambda:events.append('child_done') or 0)
        with patch.dict(os.environ, RUNPOD_POD_ID=q.POD), \
             patch.object(q.gat,'baseline',side_effect=lambda *a:events.append('baseline')), \
             patch.object(q,'process_identity',return_value='ticks'), \
             patch.object(q,'wait_for_process',side_effect=lambda *a:events.append('wait_for_ngcf')), \
             patch.object(q,'completion_check',side_effect=lambda *a:events.append('validated_done')), \
             patch.object(q.subprocess,'check_output',return_value=''), \
             patch.object(q.subprocess,'Popen',side_effect=lambda *a,**k:events.append('start_gat') or child) as start:
            q.queue('wait',7796,self.root,self.shared)
        self.assertLess(events.index('wait_for_ngcf'), events.index('validated_done'))
        self.assertLess(events.index('validated_done'), events.index('start_gat'))
        command=start.call_args.args[0]
        self.assertEqual(command[-2:], ['run','--confirmed-unstarted'])
        self.assertTrue(command[-3].endswith('hm_gat_m5.py'))
        self.assertEqual(json.loads((self.root/'queue_status.json').read_text())['phase'],'completed_gat_m5')

    def test_abnormal_ngcf_exit_never_starts_gat(self):
        with patch.dict(os.environ,RUNPOD_POD_ID=q.POD), \
             patch.object(q.gat,'baseline'), patch.object(q,'process_identity',return_value=None), \
             patch.object(q,'completion_check',side_effect=RuntimeError('not completed')), \
             patch.object(q.subprocess,'Popen') as start:
            with self.assertRaises(RuntimeError): q.queue('wait',7796,self.root,self.shared)
        start.assert_not_called()
        self.assertEqual(json.loads((self.root/'queue_status.json').read_text())['phase'],'failed_no_automatic_retry')

    def test_check_only_never_waits_or_starts_gpu_job(self):
        with patch.dict(os.environ,RUNPOD_POD_ID=q.POD), patch.object(q.gat,'baseline'), \
             patch.object(q,'process_identity',return_value='ticks'), \
             patch.object(q,'wait_for_process') as wait,patch.object(q.subprocess,'Popen') as start:
            q.queue('check',7796,self.root,self.shared)
        wait.assert_not_called();start.assert_not_called()

    def test_other_pod_is_rejected(self):
        with patch.dict(os.environ,RUNPOD_POD_ID='different'),patch.object(q.subprocess,'Popen') as start:
            with self.assertRaises(RuntimeError): q.queue('wait',7796,self.root,self.shared)
        start.assert_not_called()

    def test_process_argv_and_start_ticks(self):
        proc=Path(self.temp.name)/'proc';pid=proc/'7796';pid.mkdir(parents=True)
        args=['/workspace/clv/venv/bin/python','-u','/workspace/hm_ngcf_fixed/launch.py','run',
              '--job','hm-ngcf','--seeds','49','--models',*q.MODELS,'--confirmed-unstarted']
        (pid/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
        fields=['S']+['0']*18+['12345']+['0']*10
        (pid/'stat').write_text('7796 (python) '+' '.join(fields))
        self.assertEqual(q.process_identity(7796,proc),'12345')
        args[5]='hm-gat';(pid/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
        with self.assertRaises(RuntimeError): q.process_identity(7796,proc)

    def test_gat_baseline_and_tuple_json_identity(self):
        out=self.shared/'results/hm-gat';out.mkdir()
        (out/'runpod_invocation_1.json').write_text(json.dumps({'seeds':[49],'models':list(q.MODELS)}))
        protocol={k:False for k in storage.SCIENCE_KEYS}
        protocol.update(source_revision=g.PIN,config={'seeds':[49],'epochs':300})
        (out/'run').mkdir();(out/'run/protocol.json').write_text(json.dumps(protocol))
        (self.shared/'sources'/g.PIN).mkdir(parents=True)
        self.assertEqual(len(g.baseline(self.shared)),1)
        live=json.loads(json.dumps(protocol));live['config']['seeds']=(49,)
        storage.check_science(live,[protocol])
        artifact=out/'run/gat_m5_a_lastweek_s49.json';artifact.write_text('{}')
        with self.assertRaises(RuntimeError): g.baseline(self.shared)

    def test_gat_execute_runs_only_four_m5_with_one_preparation(self):
        calls=[];preps=[]
        prepared={'protocol':{},'data':{'splits':{'test':{}},'loss_w':None,'data_stats':{'min_item_inter':1}}}
        screen=SimpleNamespace(prepare=lambda cfg:preps.append(1) or prepared,
                               run_arm=lambda p,label:calls.append(label))
        with patch.dict('sys.modules',{'torch':SimpleNamespace()}), \
             patch.object(g,'baseline',return_value=[{}]),patch.object(g,'link_inputs'), \
             patch.object(g,'check_environment'),patch.object(storage,'check_science'), \
             patch.object(storage,'install_storage'),patch.object(g.platform,'verify_files',return_value={}), \
             patch.object(g.platform,'cuda_check'),patch.object(g.platform,'guard_space'), \
             patch.object(g.platform,'load_runner',return_value=(screen,None)), \
             patch.object(g.platform,'environment',return_value={}), \
             patch.object(g.platform,'evaluation_once',side_effect=lambda *a:nullcontext()), \
             patch.object(g.platform,'export'):
            g.execute('run',self.root,self.shared)
        self.assertEqual(preps,[1]);self.assertEqual(tuple(calls),g.MODELS)


if __name__=='__main__':unittest.main()
