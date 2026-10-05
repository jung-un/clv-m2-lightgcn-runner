"""Small CPU checks for the approved two-arm experiment; no real training data."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import clv_m2_nv_conditional_basis_screen as screen
from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN


def toy():
    # Three users, four items, undirected binary edges with degree normalization.
    edges = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 0, 0, 1, 2]])
    degree = torch.bincount(edges[0], minlength=7).float()
    weights = (degree[edges[0]] * degree[edges[1]]).rsqrt()
    return dict(n_users=3, n_items=4,
                user_q_n=np.array([.2, .9, .7]), user_q_v=np.array([.8, .8, .4]),
                user_q_c=np.array([.6, .6, .8]), user_clv_valid=np.array([True, True, False]),
                item_price_percentile=np.array([.1, .5, .9, .3]),
                item_price_valid=np.array([True, True, True, False]),
                adj=torch.sparse_coo_tensor(edges, weights, (7, 7)),
                id_dim=4, rho=.25, n_layers=2, pref_reg=.001)


class ScreenChecks(unittest.TestCase):
    def test_formula_joint_gradients_and_checkpoint(self):
        torch.manual_seed(48)
        old = M5NConditionedValueBasisLightGCN(**toy(), economic_propagation=False)
        torch.manual_seed(48)
        new = screen.ConditionalBasisLightGCN(**toy())
        for a, b in zip(old.propagated_embeddings(), new.propagated_embeddings()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        optimizer = torch.optim.Adam(new.parameters(), lr=.01)
        users, positives, negatives = torch.tensor([0, 1]), torch.tensor([2, 0]), torch.tensor([[0], [2]])
        loss, _, _ = screen.recheck._batch_loss(new, users, positives, negatives, None)
        loss.backward()
        for parameter in (new.E_u.weight, new.E_i.weight, new.T0, new.TN):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.norm() > 0
        optimizer.step()
        u, i = new.propagated_embeddings()
        uid, iid = new.id_embeddings()
        basis = new.user_value_basis
        expected = new.user_clv_level[:, None] * (
            basis @ new.T0.T + new.user_q_n_centered[:, None] * (basis @ new.TN.T))
        torch.testing.assert_close(u @ i.T, uid @ iid.T + .25 * expected @ new.item_value_basis.T)
        assert not new.economic_coordinates()[0][2].any()
        assert not new.economic_coordinates()[1][3].any()
        assert not torch.allclose(new.economic_coordinates()[0][0], new.economic_coordinates()[0][1])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "model.pt"
            torch.save(new.state_dict(), p)
            readback = screen.ConditionalBasisLightGCN(**toy())
            readback.load_state_dict(torch.load(p, weights_only=True))
            torch.testing.assert_close(new.propagated_embeddings()[0], readback.propagated_embeddings()[0])

    def test_reference_validation_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = screen.configure(out_dir=directory, baseline_json=str(Path(directory) / "m1.json"))
            metrics = {m: 1.0 for m in (*screen.ACCURACY, *screen.ECONOMIC)}
            curve = [dict(epoch=e, loss=.1, p_correct=.8,
                          **({"metrics": metrics} if e % 25 == 0 else {})) for e in range(1, 301)]
            identity = dict(seed=48, input_hash="input", source_revision="b541ba7816143bb38da9461163ea470665eb5de8")
            baseline = dict(model_id=screen.M1, curve=curve, **identity)
            payload = dict(code_version="clv-reliability-orthogonal-nv-seed48-replication-v1",
                           split=screen.SPLIT, final_test=False, holdout=False,
                           config={**asdict(cfg), "seeds": [48]}, arms=[baseline], **identity)
            Path(cfg.baseline_json).write_text(json.dumps(payload))
            _, provenance = screen.load_baseline(cfg, "input")
            with self.assertRaises(ValueError):
                screen.load_baseline(cfg, "changed-input")
            with self.assertRaises(ValueError):
                screen.validate_config(replace(cfg, epochs=100))
            with self.assertRaises(ValueError):
                screen.configure(seeds=(43,))
            arms = [baseline]
            for mid, factor in ((screen.OLD, 1.), (screen.NEW, 1.01)):
                arm = json.loads(json.dumps(baseline))
                arm["model_id"] = mid
                for r in arm["curve"]:
                    if "metrics" in r:
                        r["metrics"] = {m: value * factor for m, value in r["metrics"].items()}
                arms.append(arm)
            prep = dict(preflight={}, revision="code", input_hash="input", config_hash="config",
                        baseline_provenance=provenance)
            result = screen.report(cfg, prep, arms)
            assert result["reading"][screen.NEW]["candidate"]
            assert not result["reading"][screen.OLD]["candidate"]
            assert len(result["absolute"]) == 36
            assert all(Path(p).is_file() for p in result["paths"].values())

    def test_training_loop_and_resume(self):
        # Exercise the actual shared BPR/optimizer/checkpoint loop on a tiny graph.
        with tempfile.TemporaryDirectory() as directory, patch.object(screen.v3, "DEVICE", torch.device("cpu")):
            model = screen.ConditionalBasisLightGCN(**toy())
            cfg = replace(screen.Config(), epochs=2, eval_every=1, batch_size=2)
            prep = {"data": {"tr_u": np.array([0, 0, 1, 2]), "tr_i": np.array([0, 1, 2, 3]),
                             "pos_key": np.array([0, 1, 6, 11]), "n_items": 4}}
            identity = screen.RunIdentity("tiny", screen.NEW, 48, "config", "code", "input")
            store = screen.ProgressStore(directory, identity)
            spec = {"model_id": screen.NEW, "condition": "test"}
            metrics = {m: .1 for m in (*screen.ACCURACY, *screen.ECONOMIC)}
            with patch.object(screen.capacity, "_evaluate", return_value=metrics), patch.object(screen.capacity, "_clv_score_share", return_value={}):
                curve = screen.capacity._train_curve(model, prep, cfg, spec, 48, store)
                restored = screen.ConditionalBasisLightGCN(**toy())
                repeat = screen.capacity._train_curve(restored, prep, cfg, spec, 48, store)
            assert len(curve) == len(repeat) == 2
            assert curve[-1]["gradient_diagnostics"]["TN_gradient_norm"] > 0
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
