"""Bounded checkpoint diagnostics, no optimizer, fitting or evaluation selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import lightgcn_clv_history_linear_nv_early_stop as screen


def gradient_probe(model, users, positives, negatives, weights):
    """Gradients on the same LOO pairs; no updates, separate BPR and L2."""
    params = [(n, p) for n, p in model.named_parameters()
              if n.startswith(('n_transform.', 'v_transform.'))]
    pos, neg = model._pair_scores(users, positives, negatives)
    per_row = F.softplus(neg-pos)
    losses = [per_row.mean(), (weights*per_row).mean(),
              model.batch_l2(users, positives, negatives)]
    grads = [torch.autograd.grad(loss, [p for _, p in params], retain_graph=i < 2)
             for i, loss in enumerate(losses)]
    rows = []
    for j, (name, _) in enumerate(params):
        g0, gb, gr = [g[j].detach().flatten() for g in grads]
        bn, rn = float(gb.norm()), float(gr.norm())
        rows.append(dict(parameter=name, plain_bpr_gradient_norm=float(g0.norm()),
            weighted_bpr_gradient_norm=bn, l2_gradient_norm=rn,
            l2_to_bpr_ratio=rn/bn if bn else None,
            bpr_l2_cosine=float(torch.dot(gb, gr)/(gb.norm()*gr.norm())) if bn and rn else None,
            weighted_minus_plain_gradient_norm=float((gb-g0).norm()),
            combined_gradient_norm=float((gb+gr).norm()),
            plain_bpr=float(losses[0].detach()), weighted_bpr=float(losses[1].detach()),
            l2=float(losses[2].detach())))
    return rows


@torch.no_grad()
def score_probe(model, users, positives, negatives):
    """Exact algebraic split of LOO training-pair margins, not recommendation metrics."""
    sn, tn, sv, tv = model._axis_tables()
    an, av = model._positive_history_shares(users, positives)
    values = {}
    for axis, source, target, hist, share, q, valid, has, layer in [
        ('N',sn,tn,model.activity_history,an,model.q_n,model.activity_valid,model.n_has_history,model.n_transform),
        ('V',sv,tv,model.value_history,av,model.q_v,model.value_valid,model.v_has_history,model.v_transform)]:
        raw = model._leave_one_out(torch.sparse.mm(hist, source)[users], source[positives], share)
        mask = (valid[users].bool() & has[users] & ((1-share)>1e-8))[:,None]
        delta = target[positives]-target[negatives]
        qvalue = torch.full_like(q[users],.5) if model.constant_q else q[users]
        pieces = {'history': F.linear(raw,layer.weight[:,:-1]),
                  'q': qvalue[:,None]*layer.weight[:,-1],
                  'bias': layer.bias.expand(len(users),-1)}
        for name, component in pieces.items():
            values[f'{axis}_{name}'] = model.rho*(component*mask*delta).sum(1)
        values[f'{axis}_q_minus_constant'] = model.rho*(
            (qvalue-.5)[:,None]*layer.weight[:,-1]*mask*delta).sum(1)
    ui, ii = model._id_embeddings()
    values['ID'] = (ui[users]*(ii[positives]-ii[negatives])).sum(1)
    keys = [f'{a}_{c}' for a in ('N','V') for c in ('history','q','bias')]
    values['NV_total'] = sum(values[k] for k in keys)
    pos, neg = model._pair_scores(users, positives, negatives)
    error = float((pos-neg-values['ID']-values['NV_total']).abs().max())
    if not torch.allclose(pos-neg,values['ID']+values['NV_total'],atol=1e-5,rtol=1e-5):
        raise ValueError('Score decomposition does not reconstruct the actual LOO margin')
    return [dict(component=k,mean=float(v.mean()),mean_abs=float(v.abs().mean()),
                 std=float(v.std(unbiased=False)),positive_share=float((v>0).float().mean()),
                 reconstruction_max_error=error) for k,v in values.items()]


def run(report_paths, out_dir, batches=3, batch_size=8192, probe_seed=20260924):
    if not 1 <= batches <= 5 or not 1 <= batch_size <= 8192:
        raise ValueError('Bounded diagnostic: 1–5 batches, at most 8192 rows per batch')
    reports = [json.loads(Path(p).read_text()) for p in report_paths]
    if not reports:
        raise ValueError('Provide explicit existing result.json paths')
    cfg = screen.configure(out_dir=str(out_dir), include_m2=False)
    expected = json.loads(json.dumps(screen.preflight(cfg)))
    chosen = []
    ignored = {'seeds','out_dir','reuse_dirs','include_m2'}
    for path, report in zip(report_paths,reports):
        pf = report['preflight']
        if pf['code_version'] != screen.CODE_VERSION or pf['selection'] != expected['selection']:
            raise ValueError('Unexpected run or selection protocol')
        if pf.get('split') != expected['split'] or pf.get('final_test') is not False or pf.get('holdout') is not False:
            raise ValueError('Only the registered development split is allowed')
        for k,v in expected['config'].items():
            if k not in ignored and pf['config'].get(k) != v:
                raise ValueError(f'Protocol mismatch: {k}')
        arm = [a for a in report['arms'] if a['model_id']=='m5_linear_nv']
        if len(arm)!=1:
            raise ValueError('Each report must contain exactly one actual M5')
        a = arm[0]
        cp = Path(a['checkpoint'])
        if not cp.is_file() or screen.file_sha256(cp)!=a['checkpoint_sha256']:
            raise ValueError(f'Missing or changed checkpoint: {cp}')
        for name in ('clv_history_linear_nv_model.py','clv_history_item_fit_model.py'):
            if a['identity']['source_hashes'][name] != screen.file_sha256(Path(__file__).with_name(name)):
                raise ValueError(f'Model source mismatch: {name}')
        chosen.append((str(path),a))
    if len({a['seed'] for _,a in chosen})!=len(chosen):
        raise ValueError('Duplicate seed reports; do not silently select a run')
    # Preparation writes diagnostics only into this separate output directory.
    prep = screen.base._prepare(screen.strength_cfg(cfg))
    data=prep['data']; all_grad=[]; all_score=[]; sources=[]
    for report_path, a in chosen:
        if a['identity']['input_hash']!=prep['input_hash']:
            raise ValueError('Prepared inputs differ from the saved checkpoint')
        spec = next(s for s in screen.specs(cfg) if s['model_id']=='m5_linear_nv')
        model=screen.fixed._build(prep,screen.strength_cfg(cfg),spec,a['seed'])
        state=torch.load(a['checkpoint'],map_location='cpu',weights_only=False)
        if state['epoch']!=a['selected_epoch']:
            raise ValueError('Selected checkpoint epoch mismatch')
        model.load_state_dict(state['model_state'],strict=True); model.eval()
        rng=np.random.default_rng(probe_seed)
        for batch in range(batches):
            ix=rng.choice(len(data['tr_u']),size=min(batch_size,len(data['tr_u'])),replace=False)
            neg=screen.base.components.m4_helpers.sample_uniform_negative_matrix(
                data['tr_u'][ix],data['tr_i'][ix],data['n_items'],data['pos_key'],rng,k=1).reshape(-1)
            u,p,n=[torch.as_tensor(x,dtype=torch.long,device=screen.base.v3.DEVICE)
                   for x in (data['tr_u'][ix],data['tr_i'][ix],neg)]
            w=torch.as_tensor(prep['m4_weights'][ix],dtype=torch.float32,device=screen.base.v3.DEVICE)
            meta=dict(seed=a['seed'],selected_epoch=a['selected_epoch'],batch=batch,n_rows=len(ix))
            all_grad.extend(dict(**meta,**x) for x in gradient_probe(model,u,p,n,w))
            all_score.extend(dict(**meta,**x) for x in score_probe(model,u,p,n))
            print(f"seed {a['seed']}: diagnostic batch {batch+1}/{batches}; no parameter updates",flush=True)
        sources.append(dict(report=report_path,report_sha256=screen.file_sha256(Path(report_path)),
            seed=a['seed'],checkpoint=a['checkpoint'],checkpoint_sha256=a['checkpoint_sha256'],
            selected_epoch=a['selected_epoch']))
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,rows in [('gradients',all_grad),('score_components',all_score)]:
        paths[name]=str(root/f'{name}.csv');pd.DataFrame(rows).to_csv(paths[name],index=False)
    paths['metadata']=str(root/'diagnostic.json')
    screen.base.capacity.test10._atomic_json(Path(paths['metadata']),dict(
        sources=sources,probe_seed=probe_seed,batches=batches,batch_size=batch_size,
        scope='training-row sampled uniform-negative K1 LOO pair margins; NOT held-out or Top-K metrics',
        no_optimizer=True,no_parameter_updates=True,no_test_or_holdout=True,
        limits='Checkpoint-local gradients, not proof of training causality. Batches may overlap. Bias is item-dependent, not necessarily useless. q-minus-constant is a fixed-weight probe, not a trained control.',
        paths=paths))
    return paths
