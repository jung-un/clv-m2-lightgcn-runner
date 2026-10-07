"""Focused CPU checks for the two original-M4 A fits and six-model report."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import clv_dh_m4a_m5a_lastweek_completion as s
from test_clv_m2_nv_conditional_basis_screen import toy


def reference_fixture(metrics=None):
    cfg = s.configure()
    metrics = metrics or {m:.1 for m in (*s.common.ACCURACY,*s.common.ECONOMIC)}
    metrics.update({f"extra_{n}":.1 for n in range(142-len(metrics))})
    old_protocol = dict(
        config=asdict(cfg) | {"experiment":"dh_m3_m4b_completion", "out_dir":"old"},
        input_hash="input", split="last_seven_days_test", intervals=[1,704,711],
        validation=False, holdout=False, data_stats={}, no_early_stopping=True,
        test_evaluations_per_fit=1, test_checkpoint="fixed epoch300 only",
        m3_beta=.2, m3_audit={"coefficient_of_variation":.2})
    rows = [dict(
        model_id=model, seed=49, epochs=300, metrics=deepcopy(metrics),
        diagnostics={"rho":0.}, training_history=[{"epoch":e} for e in range(1,301)],
        identity=dict(stage="old", model_id=model, seed=49, config_hash="old",
                      source_revision="old", input_hash="input"))
        for model in (s.M1,s.M3,s.M4_B,s.M5_B)]
    reference = dict(protocol=old_protocol, rows=rows)
    prep = dict(input_hash="input", protocol=deepcopy(old_protocol))
    prep["protocol"].update(config=asdict(cfg), m4_weight_audit={"invalid_extra_absent":True})
    return cfg, reference, prep


class ACompletionChecks(unittest.TestCase):
    def test_config_formula_and_graph_loss_routing(self):
        cfg = s.configure()
        self.assertEqual(s.common.models_for(cfg),(s.M4_A,s.M5_A))
        self.assertEqual((cfg.seeds,cfg.epochs),((49,),300))
        q=np.array([.8,.8]); term=np.array([.5,.0]); valid=np.array([True,False])
        raw=s.common.previous.m5.raw_row_weights(q,q,q,term,valid,"original")
        np.testing.assert_allclose(raw,[1.2,1.0])
        with self.assertRaises(ValueError):
            s.common.configure("hm",seeds=(49,),experiment="dh_m4a_m5a_completion")
        with self.assertRaises(ValueError):
            s.common.configure(seeds=(48,),experiment="dh_m4a_m5a_completion")

        t=toy()
        data=dict(n_users=3,n_items=4,adj=t["adj"],tr_u=np.array([0,0,1,2]),
                  tr_i=np.array([0,1,2,3]),pos_key=np.array([0,1,6,11]))
        prep=dict(data=data,q_n=t["user_q_n"],q_v=t["user_q_v"],q_c=t["user_q_c"],
                  clv_valid=t["user_clv_valid"],item_amount_percentile=t["item_price_percentile"],
                  item_economic_valid=t["item_price_valid"],m3_adj=t["adj"]*.9,
                  m4_weights=np.array([.8,1.2,1.1,.9],dtype=np.float32),
                  config_hash="test",revision="code",input_hash="input")
        with tempfile.TemporaryDirectory() as d, patch.object(s.common.v3,"DEVICE",torch.device("cpu")):
            small=replace(cfg,epochs=2,id_dim=4,batch_size=2,out_dir=d)
            prep["run_dir"]=Path(d)
            m4=s.common.build_model(prep,small,s.M4_A,49)
            m5=s.common.build_model(prep,small,s.M5_A,49)
            torch.testing.assert_close(m4.E_u.weight,m5.E_u.weight,rtol=0,atol=0)
            torch.testing.assert_close(m4.adj.to_dense(),data["adj"].to_dense())
            torch.testing.assert_close(m5.adj.to_dense(),prep["m3_adj"].to_dense())
            metric={m:.1 for m in (*s.common.ACCURACY,*s.common.ECONOMIC)}
            with patch.object(s.common.fixed,"_evaluate",return_value=metric), \
                    patch.object(s.common.fixed,"_train",wraps=s.common.fixed._train) as train:
                s.common.run_arm(prep,small,s.M4_A,49)
                s.common.run_arm(prep,small,s.M5_A,49)
            self.assertEqual(train.call_count,2)
            self.assertIs(train.call_args_list[0].kwargs["row_weights"],prep["m4_weights"])
            self.assertIs(train.call_args_list[1].kwargs["row_weights"],prep["m4_weights"])

    def test_reference_and_six_model_report(self):
        cfg, reference, prep = reference_fixture()
        self.assertEqual(len(s.validate_reference(reference,prep,cfg)),4)
        changed=deepcopy(reference)
        changed["protocol"]["config"]["lr"]=.01
        with self.assertRaises(ValueError):
            s.validate_reference(changed,prep,cfg)
        changed=deepcopy(reference)
        changed["rows"][0]["training_history"].pop()
        with self.assertRaises(ValueError):
            s.validate_reference(changed,prep,cfg)

        a_rows=[]
        for model,factor in ((s.M4_A,1.01),(s.M5_A,1.02)):
            metrics={m:.1*factor for m in (*s.common.ACCURACY,*s.common.ECONOMIC)}
            metrics.update({f"extra_{n}":.1 for n in range(142-len(metrics))})
            a_rows.append(dict(model_id=model,seed=49,epochs=300,metrics=metrics,
                               diagnostics={"rho":0.},training_history=[]))
        with tempfile.TemporaryDirectory() as d:
            prep.update(run_dir=Path(d),reference_protocol=reference["protocol"],
                        reference_sha256=s.REFERENCE_SHA)
            result=s.report(prep,cfg,reference["rows"]+a_rows)
            self.assertEqual(len(result["absolute"]),6)
            self.assertEqual(len(result["comparison"]),11*142)
            self.assertEqual(len(result["interaction"]),2*142)
            self.assertTrue(result["reading"]["variants"]["A"]["descriptive_combination_condition_met"])
            self.assertFalse(result["reading"]["a_b_selected"])
            self.assertFalse(result["reading"]["significance_claim"])
            self.assertTrue(all(Path(path).is_file() for path in result["paths"].values()))


if __name__ == "__main__":
    unittest.main()
