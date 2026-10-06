"""Small CPU checks: exact reference rejection, two fits, no reference reevaluation."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import clv_dh_m3_m4b_lastweek_completion as s
from test_clv_m2_nv_conditional_basis_screen import toy


def reference_fixture():
    cfg = s.configure()
    old_cfg = asdict(s.common.configure(seeds=(49,)))
    old_cfg.pop("experiment")
    old_cfg["seeds"] = [49]
    old_cfg["out_dir"] = "/content/drive/original-run"
    old = dict(config=old_cfg, source_revision=s.REFERENCE_SOURCE, input_hash="input",
               intervals=[1,704,711], validation=False, holdout=False, data_stats={},
               split="last_seven_days_test", no_early_stopping=True, test_evaluations_per_fit=1,
               test_checkpoint="fixed epoch300 only", m4_weight_audit=dict(sha256="weights",invalid_extra_absent=True),
               m3_beta=.2, m3_audit={"coefficient_of_variation":.2})
    metrics = {m:.1 for m in (*s.common.ACCURACY,*s.common.ECONOMIC)}
    metrics.update({f"extra_{n}":.1 for n in range(142-len(metrics))})
    rows = [dict(model_id=m,seed=49,epochs=300,metrics=metrics.copy(),diagnostics={"rho":0.},
                 training_history=[{"epoch":e} for e in range(1,301)],
                 identity=dict(stage=s.common.CODE_VERSION,model_id=m,seed=49,
                               config_hash=s.REFERENCE_RUN,source_revision=s.REFERENCE_SOURCE,input_hash="input"))
            for m in (s.M1,s.M5_B)]
    prep = dict(input_hash="input",protocol=deepcopy(old))
    prep["protocol"]["config"]=asdict(cfg)
    return cfg, dict(protocol=old, rows=rows), prep


class CompletionChecks(unittest.TestCase):
    def test_reference_guards(self):
        cfg, ref, prep = reference_fixture()
        self.assertEqual(len(s.validate_reference(ref,prep,cfg)),2)
        for path, value in ((('config','lr'),.01), (('input_hash',),'wrong'),
                            (('m4_weight_audit','sha256'),'wrong'), (('m3_beta',),.3),
                            (('data_stats',),{'changed':True}), (('validation',),True)):
            changed=deepcopy(ref)
            node=changed['protocol']
            for key in path[:-1]: node=node[key]
            node[path[-1]]=value
            with self.assertRaises(ValueError): s.validate_reference(changed,prep,cfg)
        changed=deepcopy(ref)
        changed['rows'][0]['training_history'].pop()
        with self.assertRaises(ValueError): s.validate_reference(changed,prep,cfg)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): s.load_reference(Path(d)/'missing.json')
            p=Path(d)/'wrong.json'
            p.write_text('{}')
            with self.assertRaises(ValueError): s.load_reference(p)

    def test_two_actual_fits_routing_resume_and_report(self):
        t=toy()
        cfg, ref, prep = reference_fixture()
        data=dict(n_users=3,n_items=4,adj=t['adj'],tr_u=np.array([0,0,1,2]),
                  tr_i=np.array([0,1,2,3]),pos_key=np.array([0,1,6,11]))
        prep.update(data=data,q_n=t['user_q_n'],q_v=t['user_q_v'],q_c=t['user_q_c'],
                    clv_valid=t['user_clv_valid'],item_amount_percentile=t['item_price_percentile'],
                    item_economic_valid=t['item_price_valid'],m3_adj=t['adj']*.9,
                    m4_weights=np.array([.8,1.2,1.1,.9],dtype=np.float32),
                    config_hash='test',revision='code',reference_path='pinned',
                    reference_protocol=ref['protocol'],reference_sha256=s.REFERENCE_SHA)
        digest=s.hashlib.sha256(prep['m4_weights'].tobytes()).hexdigest()
        prep['protocol']['m4_weight_audit']['sha256']=digest
        with tempfile.TemporaryDirectory() as d, patch.object(s.common.v3,'DEVICE',torch.device('cpu')):
            cfg=replace(cfg,epochs=2,id_dim=4,batch_size=2,out_dir=d)
            prep.update(run_dir=Path(d))
            prep['protocol']['config']=asdict(cfg)
            refs=deepcopy(ref['rows'])
            for r in refs: r['epochs']=2
            m3=s.common.build_model(prep,cfg,s.M3,49)
            m4=s.common.build_model(prep,cfg,s.M4_B,49)
            torch.testing.assert_close(m3.E_u.weight,m4.E_u.weight,atol=0,rtol=0)
            torch.testing.assert_close(m3.adj.to_dense(),prep['m3_adj'].to_dense())
            torch.testing.assert_close(m4.adj.to_dense(),data['adj'].to_dense())
            self.assertEqual((m3.rho,m4.rho),(0.,0.))
            with patch.object(s,'configure',return_value=cfg), \
                    patch.object(s,'load_reference',return_value=ref), \
                    patch.object(s,'validate_reference',return_value=refs), \
                    patch.object(s.common.fixed,'_evaluate',return_value=refs[0]['metrics']) as evaluate, \
                    patch.object(s.common.fixed,'_train',wraps=s.common.fixed._train) as train:
                result=s.run(cfg,prep)
                self.assertEqual((train.call_count,evaluate.call_count),(2,2))
                self.assertIsNone(train.call_args_list[0].kwargs['row_weights'])
                self.assertIs(train.call_args_list[1].kwargs['row_weights'],prep['m4_weights'])
                s.run(cfg,prep)
                self.assertEqual((train.call_count,evaluate.call_count),(2,2))
            self.assertEqual(len(result['absolute']),4)
            self.assertEqual(len(result['comparison']),5*142)
            np.testing.assert_allclose(result['interaction']['interaction_absolute'],0.,atol=1e-15)
            self.assertFalse(result['reading']['descriptive_combination_condition_met'])
            self.assertFalse(result['reading']['significance_claim'])
            self.assertTrue(all(Path(p).is_file() for p in result['paths'].values()))
            changed=deepcopy(refs)
            with self.assertRaises(ValueError): s.report(prep,cfg,changed)
        with self.assertRaises(ValueError):
            s.common.configure('hm',seeds=(49,),experiment='dh_m3_m4b_completion')
        with self.assertRaises(ValueError):
            s.common.configure(seeds=(48,),experiment='dh_m3_m4b_completion')


if __name__ == '__main__':
    unittest.main()
