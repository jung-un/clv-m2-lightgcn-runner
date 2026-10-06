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
    def test_hm_three_arm_formula_training_cache_and_report(self):
        t = toy()
        data = dict(n_users=3,n_items=4,adj=t['adj'],tr_u=np.array([0,0,1,2]),
                    tr_i=np.array([0,1,2,3]),pos_key=np.array([0,1,6,11]))
        prep = dict(data=data,q_n=np.array([.2,.9,0.]),q_v=np.array([.8,.8,0.]),
                    q_c=np.array([.6,.6,0.]),clv_valid=t['user_clv_valid'],
                    item_amount_percentile=t['item_price_percentile'],
                    item_economic_valid=t['item_price_valid'],m3_adj=t['adj']*.9,
                    config_hash='hm-test',revision='code',input_hash='input')
        econ = dict(item_amount_percentile=t['item_price_percentile'],
                    user_bin_fit=np.array([[1.,.5],[.8,1.],[1.,1.]]),item_bin=np.array([0,1,0,1]))
        weights, audit = s.hm_m5.row_weights(prep,econ,check_development_reference=False)
        raw = np.array([1.1*(1+.5*.8*.1),1.1*(1+.5*.8*.5*.5),
                        1.45*(1+.5*.8*.9*.8),1.])
        np.testing.assert_allclose(weights['split_nv'],raw/raw.mean())
        with self.assertRaises(RuntimeError): s.hm_m5.row_weights(prep,econ)
        prep['m4_weights']=weights['split_nv']
        with tempfile.TemporaryDirectory() as d, patch.object(s.v3,'DEVICE',torch.device('cpu')):
            cfg=s.configure('hm',seeds=(49,),experiment='hm_m4b_m5b',out_dir=d)
            self.assertEqual((cfg.epochs,cfg.batch_size,cfg.rho),(300,131072,0.))
            self.assertEqual(s.models_for(cfg),(s.M1,s.M4_B,s.BASE))
            cfg=replace(cfg,epochs=2,id_dim=4,batch_size=2)
            prep.update(run_dir=Path(d),protocol={'config':asdict(cfg)})
            m1=s.build_model(prep,cfg,s.M1,49)
            m4=s.build_model(prep,cfg,s.M4_B,49)
            m5=s.build_model(prep,cfg,s.BASE,49)
            torch.testing.assert_close(m1.E_u.weight,m4.E_u.weight,rtol=0,atol=0)
            torch.testing.assert_close(m4.E_u.weight,m5.E_u.weight,rtol=0,atol=0)
            torch.testing.assert_close(m4.adj.to_dense(),data['adj'].to_dense())
            torch.testing.assert_close(m5.adj.to_dense(),prep['m3_adj'].to_dense())
            metrics=[{m:factor for m in (*s.ACCURACY,*s.ECONOMIC)} for factor in (1.,1.01,1.02)]
            with patch.object(s.fixed,'_evaluate',side_effect=metrics) as evaluate, \
                    patch.object(s.fixed,'_train',wraps=s.fixed._train) as train:
                result=s.run(cfg,prep)
                self.assertEqual(evaluate.call_count,3)
                self.assertIsNone(train.call_args_list[0].kwargs['row_weights'])
                self.assertIs(train.call_args_list[1].kwargs['row_weights'],prep['m4_weights'])
                self.assertIs(train.call_args_list[2].kwargs['row_weights'],prep['m4_weights'])
                s.run(cfg,prep)
                self.assertEqual((train.call_count,evaluate.call_count),(3,3))
            self.assertEqual(len(result['absolute']),3)
            self.assertTrue(result['reading']['combination_condition_met'])
            self.assertTrue(result['reading']['both_economic_at10_above_m1_and_m4_b'])
            self.assertFalse(result['reading']['interaction_attribution_claim'])
            digests=result['diagnostics'].set_index('model_id').weight_sha256
            self.assertEqual(digests[s.M4_B],digests[s.BASE])
            self.assertEqual(digests[s.BASE],audit['split_nv']['sha256'])
            self.assertTrue(result['summary']['std'].isna().all())
        with self.assertRaises(ValueError): s.configure('hm',seeds=(43,),experiment='hm_m4b_m5b')
        with self.assertRaises(ValueError): s.configure('dunnhumby',seeds=(49,),experiment='hm_m4b_m5b')

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

            # Explicit pilot executes exactly four models, not the remaining seeds.
            pilot=s.configure(out_dir=d,seeds=(49,))
            pilot_rows=[r for r in rows if r['seed']==49]
            prep['protocol']={'config':asdict(pilot)}
            with patch.object(s,'run_arm',side_effect=pilot_rows) as run_arm:
                result=s.run(pilot,prep)
            self.assertEqual(run_arm.call_count,4)
            self.assertTrue(all(call.args[-1]==49 for call in run_arm.call_args_list))
            self.assertTrue(result['reading']['pilot_only'])
            self.assertFalse(result['reading']['final_ten_seed_report'])
            self.assertEqual(result['reading']['seed_count'],1)
            self.assertTrue(result['summary']['std'].isna().all())
            self.assertTrue(result['comparison']['paired_delta_std'].isna().all())
            with self.assertRaises(RuntimeError): s.report(prep,pilot,pilot_rows[:-1])
            with self.assertRaises(ValueError): s.configure('hm',seeds=(49,))
            with self.assertRaises(ValueError): s.configure(seeds=(42,))


if __name__=="__main__":
    unittest.main()
