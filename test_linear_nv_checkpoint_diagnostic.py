import torch
from test_clv_history_linear_nv import make_model
from clv_linear_nv_checkpoint_diagnostic import gradient_probe, score_probe


def test_split_gradients_and_scores_do_not_modify_parameters():
    m=make_model()
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    u,p,n=torch.tensor([0,1]),torch.tensor([0,2]),torch.tensor([2,0])
    rows=gradient_probe(m,u,p,n,torch.tensor([1.2,.8]))
    for row in rows:
        expected=float((2*m.pref_reg*before[row['parameter']]).norm())
        assert abs(row['l2_gradient_norm']-expected)<1e-8
    scores=score_probe(m,u,p,n)
    assert max(r['reconstruction_max_error'] for r in scores)<1e-5
    for name,param in m.named_parameters():
        assert torch.equal(before[name],param)
        assert param.grad is None


def test_uniform_weights_and_constant_q_probe():
    m=make_model(constant=True)
    u,p,n=torch.tensor([0,1]),torch.tensor([0,2]),torch.tensor([2,0])
    assert all(r['weighted_minus_plain_gradient_norm']==0 for r in gradient_probe(m,u,p,n,torch.ones(2)))
    scores=score_probe(m,u,p,n)
    assert all(r['mean_abs']==0 for r in scores if 'minus_constant' in r['component'])


def test_report_to_diagnostic_and_missing_checkpoint_fail_closed(tmp_path,monkeypatch):
    import json
    import pytest
    import clv_linear_nv_checkpoint_diagnostic as d
    from test_lightgcn_clv_history_m5_strength import prepared
    prep=prepared();monkeypatch.setattr(d.screen.base.v3,'DEVICE','cpu')
    monkeypatch.setattr(d.screen.base,'_prepare',lambda cfg:prep)
    cfg=d.screen.configure(include_m2=False,out_dir=str(tmp_path/'out'))
    spec=d.screen.specs(cfg)[0]
    m=d.screen.fixed._build(prep,d.screen.strength_cfg(cfg),spec,43)
    cp=tmp_path/'selected.pt';torch.save(dict(epoch=100,model_state=m.state_dict()),cp)
    digest=d.screen.file_sha256(cp)
    arm=dict(**spec,seed=43,checkpoint=str(cp),checkpoint_sha256=digest,
        selected_epoch=100,identity=d.screen._identity(prep,cfg,spec,43))
    report=tmp_path/'result.json'
    report.write_text(json.dumps(dict(preflight=d.screen.preflight(cfg),arms=[arm])))
    paths=d.run([report],tmp_path/'diag',batches=1,batch_size=2)
    assert d.screen.file_sha256(cp)==digest
    assert json.loads(open(paths['metadata']).read())['no_parameter_updates']
    arm['checkpoint']=str(tmp_path/'missing.pt')
    report.write_text(json.dumps(dict(preflight=d.screen.preflight(cfg),arms=[arm])))
    with pytest.raises(ValueError,match='checkpoint'):
        d.run([report],tmp_path/'fail',batches=1,batch_size=2)
