"""Focused checks for final-window splitting and fixed-endpoint evaluation."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import clv_m2_m5_lastweek_test10 as s
from test_clv_m2_nv_conditional_basis_screen import toy


class LastWeekChecks(unittest.TestCase):
    def test_real_split_builder_both_datasets(self):
        for dataset in ("dunnhumby", "hm"):
            cfg = s.configure(dataset)
            base = s.base_config(cfg)
            self.assertEqual((base["VAL_DAYS"],base["TEST_DAYS"],base["HOLDOUT_DAYS"]),(0,7,0))
            entries = [(0,0,1),(0,1,698),(1,2,1),(1,3,704),
                       (2,1,697),(2,2,704),(0,2,705),(0,0,711),(1,1,711)]
            frame = pd.DataFrame(entries,columns=["u_raw","i_raw","t"])
            frame["v"], frame["up"] = 1., 1.
            if dataset == "hm":
                frame["t"] = frame.t.map(lambda x: pd.Timestamp("2018-09-20") if x==1
                                         else pd.Timestamp("2020-09-22")-pd.Timedelta(days=711-x))
                frame["i_raw"] = frame.i_raw.astype(str)
            else:
                frame["b_raw"] = np.arange(len(frame))
            dcfg = s.v3.DCFG.copy()
            meta = pd.DataFrame({dcfg["item_key_col"]:frame.i_raw.unique(),
                                 dcfg["category_col"]:["a","b","a","b"]})
            with patch.object(s.v3,"load_transactions",return_value=frame), \
                    patch.object(s.v3.pd,"read_csv",return_value=meta), \
                    patch.object(s.v3,"DEVICE",torch.device("cpu")):
                data = s.v3.prepare_data(base,dcfg)
            s.validate_split(data,cfg)
            self.assertEqual(len(data["train"]),6)
            truth,_ = data["splits"]["test"]
            self.assertEqual(sum(map(len,truth.values())),2)  # repeated (0,0) excluded
            keys = np.concatenate([u*data["n_items"]+items for u,items in truth.items()])
            self.assertFalse(np.isin(keys,data["pos_key"]).any())

    def test_training_once_then_evaluation_and_cache(self):
        t = toy()
        data = dict(n_users=3,n_items=4,adj=t["adj"],tr_u=np.array([0,0,1,2]),
                    tr_i=np.array([0,1,2,3]),pos_key=np.array([0,1,6,11]))
        prep = dict(data=data,q_n=t["user_q_n"],q_v=t["user_q_v"],q_c=t["user_q_c"],
                    clv_valid=t["user_clv_valid"],item_amount_percentile=t["item_price_percentile"],
                    item_economic_valid=t["item_price_valid"],m3_adj=t["adj"],
                    m4_weights=np.array([.8,1.2,1.1,.9]),config_hash="test",revision="code",input_hash="input")
        metrics = {m:.1 for m in (*s.ACCURACY,*s.ECONOMIC)}
        with tempfile.TemporaryDirectory() as d, patch.object(s.v3,"DEVICE",torch.device("cpu")):
            cfg=replace(s.configure(out_dir=d),epochs=2,id_dim=4,batch_size=2)
            prep["run_dir"]=Path(d)
            with patch.object(s.fixed,"_evaluate",return_value=metrics) as evaluate, \
                    patch.object(s.fixed,"_train",wraps=s.fixed._train) as train:
                row=s.run_arm(prep,cfg,s.FULL,42)
                self.assertEqual(len(row["training_history"]),2)
                self.assertTrue(all("metrics" not in r for r in row["training_history"]))
                evaluate.assert_called_once()
                np.testing.assert_array_equal(train.call_args.kwargs["row_weights"],prep["m4_weights"])
                self.assertGreater(row["diagnostics"]["TN_gradient_norm"],0)
                self.assertEqual(s.run_arm(prep,cfg,s.FULL,42),row)
                evaluate.assert_called_once()  # cached fit must not reevaluate test
            m1=s.build_model(prep,cfg,s.M1,42)
            self.assertEqual(m1.rho,0)
            self.assertFalse(m1.economic_propagation)
            store=s.ProgressStore(Path(d)/"progress",s.RunIdentity(s.CODE_VERSION,s.FULL,42,"test","code","input"))
            restored=s.build_model(prep,cfg,s.FULL,42)
            history=s.fixed._train(restored,prep,cfg,s.FULL,42,store,row_weights=prep["m4_weights"])
            self.assertEqual(len(history),2)
            np.testing.assert_allclose(restored.TN.detach(),row["diagnostics"]["TN"])

    def test_all_ten_seeds_required_and_paired_mean(self):
        rows=[]
        for seed in s.SEEDS:
            for model,factor in ((s.M1,1.),(s.BASE,1.01),(s.M2,.97),(s.FULL,1.02)):
                rows.append(dict(model_id=model,seed=seed,diagnostics={},
                                 metrics={m:factor*(seed-40) for m in (*s.ACCURACY,*s.ECONOMIC)}))
        with tempfile.TemporaryDirectory() as d:
            cfg=s.configure(out_dir=d)
            prep=dict(run_dir=Path(d),protocol={"config":asdict(cfg)})
            with self.assertRaises(RuntimeError): s.report(prep,cfg,rows[:-1])
            result=s.report(prep,cfg,rows)
            self.assertTrue(result["reading"]["combination_condition_met"])
            self.assertEqual(len(result["absolute"]),40)
            self.assertTrue(result["summary"]["count"].eq(10).all())
            comparison=result["comparison"]
            x=comparison[(comparison.model_id==s.FULL)&(comparison.reference==s.BASE)].iloc[0]
            self.assertAlmostEqual(x.relative_change_pct,100*(1.02/1.01-1))
            self.assertEqual(x.seeds_improved,10)
            self.assertTrue(all(Path(p).is_file() for p in result["paths"].values()))


if __name__=="__main__":
    unittest.main()
