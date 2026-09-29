"""One CPU check for the M2+M3 joint forward and plain BPR boundary."""
import numpy as np
import pandas as pd
import torch

import clv_shared_feature_centered_graph_screen as screen
from clv_shared_feature_residual_m2 import AXES, build_features
from test_lightgcn_clv_history_m5_strength import prepared


def test_joint_graph_and_nv_gradient(tmp_path, monkeypatch):
    monkeypatch.setattr(screen.m2.base.v3, 'DEVICE', 'cpu')
    prep = prepared()
    data = prep['data']
    train = pd.DataFrame(dict(u_idx=data['tr_u'], i_idx=data['tr_i'],
                              up=[1., 3., 2., 4.], cat_idx=[0, 0, 0, 0]))
    features = build_features(train, n_users=2, n_items=data['n_items'],
        q_n=prep['q_n'], q_v=prep['q_v'], n_valid=[True, True], v_valid=[True, True])
    weighted = screen.m2.base.v3.build_adj(data['tr_u'], data['tr_i'],
        np.array([1.3, .7, 1.2, .8], np.float32), 2, data['n_items'])
    prep.update(features=features, weighted_adj=weighted)
    cfg = screen.m2.configure(str(tmp_path))
    model = screen._build(prep, cfg)
    assert not torch.equal(model.adj.values(), data['adj'].coalesce().values())
    assert screen._spec()['weighted'] is False  # No M4 row weights.
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 2])
    negatives = torch.tensor([3, 4])
    loss, _ = model.bpr_loss(users, positives, negatives)
    loss.backward()
    assert all(model.encoders[name].weight.grad.norm() > 0 for name in AXES)
    try:
        model.bpr_loss(users, positives, negatives, weights=torch.ones(2))
    except ValueError:
        pass
    else:
        raise AssertionError('M4 row weights unexpectedly accepted')
