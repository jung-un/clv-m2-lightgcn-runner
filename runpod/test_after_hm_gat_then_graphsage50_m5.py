import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import after_hm_gat_then_graphsage50_m5 as q

class Gate(unittest.TestCase):
    def test_requires_all_four_complete_results(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'run'; (p/'arms').mkdir(parents=True); (p/'progress/stages').mkdir(parents=True)
            proto=dict(source_revision=q.platform.JOBS['hm-gat']['pin'],config={'epochs':300,'seeds':[49]},
                       code_version='stage',input_hash='hash',validation=False,holdout=False,early_stopping=False)
            (p/'protocol.json').write_text(json.dumps(proto))
            for label in ('M1','M2','M3','M4'):
                mid=f'gat_{label.lower()}_lastweek'
                ident=dict(stage='stage',model_id=mid,seed=49,config_hash='run',source_revision=proto['source_revision'],input_hash='hash')
                (p/'arms'/f'{mid}_s49.json').write_text(json.dumps(dict(identity=ident,model=label,seed=49,epochs=300,
                    metrics={str(i):1.0 for i in range(142)},training_history=[{'epoch':i} for i in range(1,301)])))
                (p/'progress/stages'/f'stage_{mid}_s49.completed.json').write_text(json.dumps(dict(identity=ident,status='completed')))
            q.completion_check(p)
            marker=p/'progress/stages/stage_gat_m4_lastweek_s49.completed.json'
            marker.write_text('{}')
            with self.assertRaises(RuntimeError):q.completion_check(p)
    def test_check_never_launches(self):
        with TemporaryDirectory() as d, patch.object(q,'SHARED',Path(d)),patch.object(q,'preflight'), \
             patch.object(q,'process_identity',return_value='ticks'),patch.object(q.subprocess,'Popen') as child:
            q.queue('check',1945)
            child.assert_not_called()
    def test_wrong_pod_refused(self):
        with patch.dict(q.os.environ,{'RUNPOD_POD_ID':'wrong'}):
            with self.assertRaises(RuntimeError):q.preflight()

if __name__=='__main__':unittest.main()
