from dataclasses import replace
import pytest
import torch

import clv_hm_gcn_final8_seed50 as hm
import test_clv_dh_gcn_final8_seed49 as dh_tests
from test_clv_m5_ab_adaptive_lastweek_seed49 import prepared as reference_fixture
from test_clv_dh_gcn_final8_seed49 import prepared


@pytest.mark.parametrize('label', hm.ORDER)
def test_hm_arms_match_fixed_gcn_and_loss_routing(prepared, label, monkeypatch):
    monkeypatch.setattr(hm.base.v3, 'DEVICE', torch.device('cpu'))
    prepared['cfg'] = hm.configure(out_dir=str(prepared['run_dir']))
    monkeypatch.setattr(dh_tests, 's', hm)
    dh_tests.test_eight_arms_routing_and_joint_gradients(prepared, label)


def test_config_and_common_initialization(prepared, monkeypatch):
    monkeypatch.setattr(hm.base.v3, 'DEVICE', torch.device('cpu'))
    cfg = hm.configure(out_dir='/tmp/gcn-new-output', m2_reference='/tmp/ref2', m4_reference='/tmp/ref4')
    hm.validate_config(cfg)
    assert cfg.dataset == 'hm' and cfg.seeds == (50,) and cfg.epochs == 300
    assert cfg.batch_size == 131072 and cfg.negative_count == 1
    for wrong in (replace(cfg, epochs=299), replace(cfg, seeds=(49,)), replace(cfg, lr=.01)):
        with pytest.raises(ValueError):
            hm.validate_config(wrong)
    prepared['cfg'] = cfg
    monkeypatch.setattr(dh_tests, 's', hm)
    dh_tests.test_common_initialization_and_dense_gcn_formula(prepared)


def test_hm_identity_does_not_reuse_dh_or_ngcf(prepared):
    prepared['cfg'] = hm.configure(out_dir=str(prepared['run_dir']))
    ident = hm.identity(prepared, 'M5_D')
    assert ident.stage == 'clv-hm-gcn-final8-seed50-v1'
    assert ident.model_id == 'gcn_m5_d_lastweek' and ident.seed == 50

