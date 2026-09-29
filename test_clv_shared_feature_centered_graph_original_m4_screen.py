"""Small CPU check: only the M5 combination may use original M4 row weights."""
import numpy as np
import pandas as pd
import pytest
import torch

import clv_shared_feature_centered_graph_original_m4_screen as screen
from clv_shared_feature_residual_m2 import AXES, build_features
from test_lightgcn_clv_history_m5_strength import prepared


def test_joint_weighted_bpr_and_nv_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr(screen.joint.m2.base.v3, 'DEVICE', 'cpu')
    prep = prepared()
    data = prep['data']
    train = pd.DataFrame(dict(u_idx=data['tr_u'], i_idx=data['tr_i'],
                              up=[1., 3., 2., 4.], cat_idx=[0, 0, 0, 0]))
    prep['features'] = build_features(train, n_users=2, n_items=data['n_items'],
        q_n=prep['q_n'], q_v=prep['q_v'], n_valid=[True, True], v_valid=[True, True])
    prep['weighted_adj'] = screen.joint.m2.base.v3.build_adj(data['tr_u'], data['tr_i'],
        np.array([1.3, .7, 1.2, .8], np.float32), 2, data['n_items'])
    model = screen._build(prep, screen.joint.m2.configure(str(tmp_path)))
    users = torch.tensor([0, 0, 1, 1])
    positives = torch.tensor([0, 1, 1, 2])
    negatives = torch.tensor([[3], [4], [3], [4]])
    weights = torch.tensor([.7, 1.3, .8, 1.2])
    loss, _, _ = screen.joint.m2.base.components._batch_loss(
        model, users, positives, negatives, weights)
    pos, neg = model._pair_scores(users, positives, negatives[:, 0])
    expected = (weights*torch.nn.functional.softplus(neg-pos)).mean() + \
        model.batch_l2(users, positives, negatives[:, 0])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(model.encoders[name].weight.grad.norm() > 0 for name in AXES)
    assert screen._spec()['weighted'] is True
    with pytest.raises(ValueError):
        model.weighted_bpr_loss(users, positives, negatives[:, 0], torch.tensor([1., 0., 1., 1.]))
