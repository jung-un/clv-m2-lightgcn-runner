"""Minimal checks for paired movement arithmetic and unchanged populations."""
import unittest

import numpy as np
import pandas as pd

import clv_m5_nv_rank_diagnostic as diag


class RankDiagnosticTest(unittest.TestCase):
    def test_cutoff_movements_and_missing_rank(self):
        users = pd.DataFrame(dict(seed=[43, 43], user=[0, 1], truth_count=[2, 1],
            segment=["중CLV", "고CLV"], clv_valid=[True, True], q_n=[.2, .8], q_v=[.7, .3],
            n_bin=[1, 5], v_bin=[4, 2], degree_bin=[1, 1]))
        base = pd.DataFrame(dict(seed=[43]*3, user=[0, 0, 1], item=[3, 4, 5],
                                weight=[2., 5., 4.], rank=[20., 60., 10.]))
        candidate = base.assign(rank=[21., 50., np.nan])
        truth, per_user = diag.movement_tables(base, candidate, users)
        self.assertEqual(truth.loc[2, "m5_band"], ">100")
        self.assertEqual(per_user["net_recall@20"].mean(), -.75)
        self.assertEqual(per_user["net_weight@20"].mean(), -3.)
        self.assertEqual(per_user["net_recall@50"].mean(), -.25)
        self.assertEqual(per_user["net_weight@50"].mean(), .5)
        summary, correlations, transitions = diag.summaries(truth, per_user)
        row = summary[summary.grouping.eq("all") & summary.k.eq(20)].iloc[0]
        self.assertEqual(row.truth_lost, 2)
        self.assertEqual(row.truth_gained, 0)
        self.assertEqual(transitions.truth_count.sum(), 3)
        self.assertEqual(len(correlations), 6)
        with self.assertRaises(ValueError):
            diag.movement_tables(base, candidate.iloc[:2], users)

    def test_train_bins_keep_ties_and_invalid_values(self):
        bins, _ = diag.train_bins([0, 0, .5, 1, np.nan], [1, 1, 1, 1, 0],
                                 np.arange(5))
        self.assertEqual(bins[0], bins[1])
        self.assertEqual(bins[4], 0)
        self.assertLess(bins[0], bins[3])

    def test_full_catalog_score_masks_train_pair_and_records_margin(self):
        from types import SimpleNamespace
        import torch
        import lightgcn_clv_v3 as v3
        v3.DEVICE = torch.device("cpu")

        class Model(torch.nn.Module):
            def embeddings(self):
                return (torch.ones(1, 1), torch.arange(120., 0., -1.).reshape(-1, 1),
                        torch.zeros(1, 1), torch.zeros(120, 1))

        prepared = dict(data=dict(n_users=1, n_items=120, csr_ptr=np.array([0, 1]),
            csr_items=np.array([0])), cache=SimpleNamespace(users=np.array([0]),
            gt={0: np.array([20, 119])}, rev={0: np.array([2., 3.])}),
            base_cfg={"EVAL_BATCH": 32})
        result = diag._score_truth(Model(), prepared, 43)
        self.assertEqual(result.loc[0, "rank"], 20)
        self.assertEqual(result.loc[0, "margin@20"], 0.)
        self.assertTrue(np.isnan(result.loc[1, "rank"]))
        self.assertLess(result.loc[1, "margin@50"], 0.)
        prepared["cache"].gt[0] = np.array([0])
        with self.assertRaises(RuntimeError):
            diag._score_truth(Model(), prepared, 43)


if __name__ == "__main__":
    unittest.main()
