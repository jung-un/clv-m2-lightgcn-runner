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
        recommendations = []
        result = diag._score_truth(Model(), prepared, 43, recommendations)
        self.assertEqual(len(recommendations), 50)
        self.assertNotIn(0, [r["item"] for r in recommendations])
        self.assertEqual(sum(r["is_truth"] for r in recommendations), 1)
        self.assertEqual(result.loc[0, "rank"], 20)
        self.assertEqual(result.loc[0, "margin@20"], 0.)
        self.assertTrue(np.isnan(result.loc[1, "rank"]))
        self.assertLess(result.loc[1, "margin@50"], 0.)
        prepared["cache"].gt[0] = np.array([0])
        with self.assertRaises(RuntimeError):
            diag._score_truth(Model(), prepared, 43)

    def test_train_candidate_features_and_same_user_differences(self):
        from clv_m1_m5_error_diagnostic import attach_features, enrich_tables
        users = pd.DataFrame(dict(seed=[43], user=[0], segment=["고CLV"], q_n=[.9],
            q_v=[.8], clv_valid=[True], degree=[2], truth_count=[2]))
        train = pd.DataFrame(dict(u_idx=[0, 0, 1], i_idx=[0, 1, 2], v=[2., 6., 8.]))
        prep = dict(data=dict(train=train, n_items=120, item_cat=np.arange(120) % 2),
            item_amount_percentile=np.linspace(0, 1, 120), item_economic_valid=np.ones(120, bool))
        candidates = pd.DataFrame(dict(seed=[43, 43], user=[0, 0], item=[2, 3]))
        features = attach_features(candidates, users, prep)
        np.testing.assert_allclose(features.category_row_share, [.5, .5])
        np.testing.assert_allclose(features.category_spend_share, [.25, .75])
        self.assertEqual(features.item_buyers.tolist(), [1, 0])
        np.testing.assert_allclose(features.user_amount_position, [.75/119]*2)
        truth = pd.DataFrame(dict(seed=[43, 43], user=[0, 0], item=[60, 61],
                                 rank_m3=[np.nan, np.nan], rank_m5=[np.nan, np.nan]))
        recs = pd.DataFrame([dict(seed=43, user=0, item=i+2, rank=i+1,
            is_truth=False, model=label) for label in ("m1", "m5") for i in range(50)])
        tables = enrich_tables(truth, users, recs, prep)
        self.assertEqual(len(tables['truth_features']), 2)
        self.assertEqual(len(tables['paired_user_feature_differences']), 6)
        self.assertTrue(tables['truth_features']['status@10'].eq('both_miss').all())
        bad = recs.copy()
        bad.loc[0, 'is_truth'] = True
        with self.assertRaises(RuntimeError):
            enrich_tables(truth, users, bad, prep)

    def test_baseline_identity_and_missing_checkpoint_are_rejected(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        from clv_m1_m5_error_diagnostic import baseline_source
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'results_v3_dunnhumby_clv_m2_capacity_search_v1'
            report_path = folder / 'arms/hash/baseline_m1_bpr_k1_s43.json'
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps(dict(id_dim=64, pref_reg=.001, seed=43,
                condition='baseline', source_revision='revision', curve=[dict(epoch=300, metrics={})])))
            cfg = SimpleNamespace(id_dim=64, pref_reg=.001)
            with self.assertRaises(RuntimeError):
                baseline_source(root, 43, cfg, {'input_hash': 'input'})
            ckpt = folder / 'progress/hash/resume/capacity_search_dev_baseline_m1_bpr_k1_s43_latest.pt'
            ckpt.parent.mkdir(parents=True)
            identity = dict(stage='capacity_search_dev', model_id='baseline_m1_bpr_k1', seed=43,
                            config_hash='hash', source_revision='revision', input_hash='WRONG')
            torch.save(dict(epoch=300, identity=identity, model_state={}), ckpt)
            spec, _ = baseline_source(root, 43, cfg, {'input_hash': 'input'})
            self.assertEqual(spec[1], 'm1_bpr_k1')
            with self.assertRaises(RuntimeError):
                diag._checkpoint(folder, identity['stage'], identity['model_id'], 43,
                                 'input', 'hash', 'revision')
            identity['input_hash'] = 'input'
            torch.save(dict(epoch=300, identity=identity, model_state={}), ckpt)
            state, _ = diag._checkpoint(folder, identity['stage'], identity['model_id'], 43,
                                       'input', 'hash', 'revision')
            self.assertEqual(state, {})

    def test_hm_features_limit_customer_context_but_keep_full_item_buyers(self):
        from clv_hm_m1_m5_error_diagnostic import attach_hm_features
        users = pd.DataFrame(dict(seed=[43], user=[0], segment=['고CLV'], q_n=[.9],
            q_v=[.8], clv_valid=[True], degree=[2], truth_count=[1]))
        train = pd.DataFrame(dict(u_idx=[0, 0, 0, 1, 1], i_idx=[0, 0, 1, 0, 2],
                                  v=[1., 1., 6., 4., 8.]))
        prep = dict(data=dict(train=train, n_items=4,
            tr_u=np.array([0, 0, 0, 1, 1]), tr_i=np.array([0, 0, 1, 0, 2]),
            pos_key=np.array([0, 1, 4, 6])),
            item_cat=np.array([0, 1, 0, 1]),
            item_amount_percentile=np.array([.2, .8, .6, .4]),
            item_economic_valid=np.ones(4, bool))
        candidates = pd.DataFrame(dict(seed=[43, 43], user=[0, 0], item=[0, 2]))
        actual = attach_hm_features(candidates, users, prep)
        self.assertEqual(actual.item_buyers.tolist(), [2, 1])
        np.testing.assert_allclose(actual.category_row_share, [2 / 3, 2 / 3])
        np.testing.assert_allclose(actual.category_spend_share, [.25, .25])
        np.testing.assert_allclose(actual.user_amount_position, [.65, .65])
        self.assertEqual(prep['_hm_candidate_feature_context']['filtered_train_rows'], 3)

    def test_hm_compact_checkpoint_identity_is_strict(self):
        import tempfile
        from pathlib import Path
        import torch
        from clv_hm_m1_m5_error_diagnostic import _load_compact_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'progress/hash/resume/checkpoint_latest.pt'
            path.parent.mkdir(parents=True)
            identity = dict(stage='stage', model_id='model', seed=43,
                            config_hash='arm', source_revision='revision',
                            input_hash='wrong')
            torch.save(dict(epoch=300, identity=identity,
                            parameter_state={'weight': torch.ones(1)}), path)
            with self.assertRaises(RuntimeError):
                _load_compact_checkpoint(folder, 'progress/*/resume/*_latest.pt',
                    stage='stage', model_id='model', input_hash='input',
                    source_revision='revision')
            identity['input_hash'] = 'input'
            torch.save(dict(epoch=300, identity=identity,
                            parameter_state={'weight': torch.ones(1)}), path)
            state, source = _load_compact_checkpoint(
                folder, 'progress/*/resume/*_latest.pt', stage='stage',
                model_id='model', input_hash='input', source_revision='revision')
            self.assertIn('weight', state)
            self.assertEqual(source['epoch'], 300)

    def test_hm_readback_only_relaxes_tie_sensitive_alignment(self):
        from clv_hm_m1_m5_error_diagnostic import ALIGNMENT, _readback_match
        self.assertTrue(_readback_match(ALIGNMENT, .078048, .078044)[0])
        self.assertFalse(_readback_match('recall@10', .078048, .078044)[0])
        self.assertFalse(_readback_match(ALIGNMENT, .079044, .078044)[0])


if __name__ == "__main__":
    unittest.main()
