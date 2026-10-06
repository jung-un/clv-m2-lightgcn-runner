import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

import clv_m2_direct_nv_lastweek as s
from clv_run_state import ProgressStore, RunIdentity


def empty_adj(n_users=2, n_items=2):
    return torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long), torch.empty(0),
        (n_users + n_items, n_users + n_items),
    ).coalesce()


def model(active=True):
    return s.DirectNVM2(
        n_users=2, n_items=2,
        user_q_n=np.array([.9, .1]), user_q_v=np.array([.5, .5]),
        user_clv_valid=np.array([True, True]),
        item_buyer_q_n=np.array([.5, .9]),
        item_amount_percentile=np.array([.5, .5]),
        item_n_sum=np.array([1., .9]), item_n_count=np.array([2., 1.]),
        active=active, adj=empty_adj(), id_dim=3, n_layers=0, pref_reg=0.,
    )


class DirectNVChecks(unittest.TestCase):
    def test_direct_coordinates_stay_outside_graph_and_gamma_learns(self):
        m = model()
        with torch.no_grad():
            m.E_u.weight.zero_()
            m.E_i.weight.zero_()
        user, item = m.propagated_embeddings()
        torch.testing.assert_close(user[:, 3:], torch.tensor([[.8, 0.], [-.8, 0.]]))
        torch.testing.assert_close(item[:, 3:], torch.tensor([[0., 0.], [.8, 0.]]))
        loss, diagnostics = m.bpr_loss(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([1])
        )
        # Positive item0 excludes user0: remaining buyer q_N=.1, hence margin=-1.28.
        torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(1.28)))
        loss.backward()
        self.assertGreater(float(m.gamma.grad.norm()), 0.)
        self.assertLess(diagnostics["p_correct"], 1.)
        self.assertFalse(m.representation_diagnostics()["economic_graph_propagation"])

    def test_m1_is_exact_nonintervention_and_report_requires_both_arms(self):
        torch.manual_seed(7)
        m1 = model(active=False)
        torch.manual_seed(7)
        m2 = model(active=True)
        torch.testing.assert_close(m1.E_u.weight, m2.E_u.weight, rtol=0, atol=0)
        self.assertEqual(float(m1.economic_coordinates()[0].abs().sum()), 0.)
        metrics = {name: 1. for name in (*s.ACCURACY, *s.ECONOMIC)}
        with tempfile.TemporaryDirectory() as directory:
            cfg = s.configure(out_dir=directory)
            prep = {"run_dir": Path(directory), "protocol": {}}
            rows = [
                {"model_id": s.M1, "seed": 49, "metrics": metrics, "diagnostics": {}},
                {"model_id": s.M2, "seed": 49,
                 "metrics": {name: 1.01 for name in metrics}, "diagnostics": {}},
            ]
            with self.assertRaises(RuntimeError):
                s.report(prep, cfg, rows[:1])
            result = s.report(prep, cfg, rows)
            self.assertTrue(result["reading"]["candidate_condition_met"])
            self.assertTrue(result["reading"]["pilot_only"])
            self.assertTrue(all(Path(path).is_file() for path in result["paths"].values()))
        with self.assertRaises(ValueError):
            s.configure(seeds=(42,))

    def test_shared_training_loop_updates_direct_m2(self):
        m = model()
        prep = {"data": {"tr_u": np.array([0, 1]), "tr_i": np.array([0, 1]),
                         "pos_key": np.array([0, 3]), "n_items": 2}}
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(s.configure(out_dir=directory), epochs=2, batch_size=2)
            identity = RunIdentity(s.CODE_VERSION, s.M2, 49, "test", "code", "input")
            history = s.fixed._train(
                m, prep, cfg, s.M2, 49, ProgressStore(Path(directory), identity)
            )
        self.assertEqual([row["epoch"] for row in history], [1, 2])
        self.assertFalse(torch.equal(m.gamma.detach(), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
