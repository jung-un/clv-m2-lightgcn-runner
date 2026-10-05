"""Targeted CPU check: routing, weighted joint training/resume, reporting guards."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import clv_m2_nv_conditional_basis_m5_screen as s
from test_clv_m2_nv_conditional_basis_screen import toy


class CombinationCheck(unittest.TestCase):
    def test_combination_flow(self):
        cfg = s.configure()
        s.validate_config(cfg)
        with self.assertRaises(ValueError):
            s.validate_config(replace(cfg, seeds=(43,)))
        metrics = {m: 1. for m in (*s.ACCURACY, *s.ECONOMIC)}
        old = pd.DataFrame([dict(model_id=mid, seed=44, epoch=e, **metrics)
                            for mid in (s.M1, s.M5_BASE) for e in range(25, 301, 25)])
        s.validate_curve(old, (s.M1, s.M5_BASE), cfg)
        with self.assertRaises(ValueError):
            s.validate_curve(pd.concat([old, old.iloc[:1]]), (s.M1, s.M5_BASE), cfg)
        metadata = dict(code_version=s.m5.CODE_VERSION, seed=44, input_hash="input",
                        source_revision="9814a7bae305a1d6257fd36be4410afcbe854372",
                        split=s.basis.SPLIT, final_test=False, holdout=False,
                        config=json.loads(json.dumps(asdict(cfg))), beta=.216, **{"lambda": .5},
                        weight_audit={s.M5_BASE: {"sha256": "weights"}})
        s.validate_reference(metadata, cfg, "input", {"beta": .216}, {"sha256": "weights"})
        with self.assertRaises(ValueError):
            s.validate_reference(metadata, cfg, "input", {"beta": .216}, {"sha256": "wrong"})
        with self.assertRaises(ValueError):
            s.validate_reference(metadata | {"final_test": True}, cfg, "input",
                                 {"beta": .216}, {"sha256": "weights"})
        t = toy()
        prep = dict(data=dict(n_users=3, n_items=4, adj=t["adj"],
                              tr_u=np.array([0, 0, 1, 2]), tr_i=np.array([0, 1, 2, 3]),
                              pos_key=np.array([0, 1, 6, 11])),
                    q_n=t["user_q_n"], q_v=t["user_q_v"], q_c=t["user_q_c"],
                    clv_valid=t["user_clv_valid"], item_amount_percentile=t["item_price_percentile"],
                    item_economic_valid=t["item_price_valid"],
                    signals=dict(edge_users=np.array([0, 0, 1, 2]), edge_items=np.array([0, 1, 2, 3])),
                    m3_graph={"weights": np.array([.5, 1.5, 1., 1.])},
                    m4_weights=np.array([.8, 1.2, 1.1, .9]), config_hash="config",
                    revision="source", input_hash="input", old_curve=old,
                    baseline_provenance={})
        with tempfile.TemporaryDirectory() as directory, patch.object(s.v3, "DEVICE", torch.device("cpu")):
            cfg = s.configure(out_dir=directory)
            tiny = replace(cfg, epochs=2, eval_every=1, id_dim=4, batch_size=2)
            plain, full = s.build_model(prep, tiny, s.M2), s.build_model(prep, tiny, s.M5_FULL)
            torch.testing.assert_close(plain.E_u.weight, full.E_u.weight, rtol=0, atol=0)
            self.assertFalse(torch.allclose(plain.adj.to_dense(), full.adj.to_dense()))
            self.assertFalse(full.economic_propagation)
            with patch.object(s.capacity, "_evaluate", return_value=metrics), \
                    patch.object(s.capacity, "_clv_score_share", return_value={}), \
                    patch.object(s.capacity, "_train_curve", wraps=s.capacity._train_curve) as train:
                for mid in (s.M2, s.M5_FULL):
                    arm = s.run_arm(tiny, prep, mid)
                    self.assertGreater(arm["final_diagnostics"]["TN_gradient_norm"], 0)
                    passed_weights = train.call_args.kwargs["row_weights"]
                    if mid == s.M2:
                        self.assertIsNone(passed_weights)
                    else:
                        np.testing.assert_array_equal(passed_weights, prep["m4_weights"])
                    self.assertEqual(s.run_arm(tiny, prep, mid)["identity"], arm["identity"])
                identity = s.RunIdentity(s.CODE_VERSION, s.M5_FULL, 44, "config", "source", "input")
                store = s.ProgressStore(Path(directory) / "progress/config", identity)
                restored = s.build_model(prep, tiny, s.M5_FULL)
                history = s.capacity._train_curve(restored, prep, tiny, {"model_id": s.M5_FULL,
                    "condition": "test"}, 44, store, row_weights=prep["m4_weights"])
                self.assertEqual(len(history), 2)
                np.testing.assert_allclose(restored.TN.detach(), arm["final_diagnostics"]["TN"])
            arms = [dict(model_id=mid, curve=[dict(epoch=e, loss=.1, p_correct=.8,
                    metrics={m: v*factor for m,v in metrics.items()}) for e in range(25,301,25)])
                    for mid,factor in ((s.M2,.97), (s.M5_FULL,1.02))]
            prep["preflight"] = {"config": asdict(cfg)}
            # Standalone failure must neither block training nor fail a successful combination.
            with patch.object(s.m5, "_existing_m5_b_curves", return_value=(old, {})), \
                    patch.object(s, "run_arm", side_effect=arms) as fits:
                result = s.run(cfg, prep)
                self.assertEqual([c.args[-1] for c in fits.call_args_list], [s.M2, s.M5_FULL])
            self.assertTrue(result["reading"]["combination_candidate"])
            self.assertEqual(len(result["absolute"]), 48)
            self.assertTrue(all(Path(p).is_file() for p in result["paths"].values()))
            at = result["absolute"].copy()
            at.loc[at.model_id.eq(s.M5_BASE), list(s.ECONOMIC)] = 1.03
            self.assertFalse(s.reading(at)["combination_candidate"])  # M1-only win is insufficient.
            at = result["absolute"].copy()
            at.loc[at.model_id.eq(s.M5_FULL), "recall@50"] = .98
            self.assertFalse(s.reading(at)["combination_candidate"])


if __name__ == "__main__":
    unittest.main()
