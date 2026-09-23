"""Staged development comparison with identical chronological early stopping.

The fixed-100 runner is intentionally preserved. Historical curves are replayed
only up to the first stop; isolated epoch-100 results are not selected baselines.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_history_linear_nv as fixed
from clv_run_state import ProgressStore, RunIdentity, clone_state, _atomic_torch, file_sha256

base = fixed.base
CODE_VERSION = "history-linear-nv-early-stop-development-v2"
MONITOR = fixed.PRIMARY[0]


@dataclass(frozen=True)
class Config(base.StrengthConfig):
    m4_mode: str = "complementary"
    seeds: tuple = (43,)
    rhos: tuple = (0.05,)
    epochs: int = 300
    eval_every: int = 25
    patience_start: int = 100
    patience: int = 4
    include_controls: bool = False


def configure(**overrides):
    defaults = dict(out_dir=f"{base.v3.default_out_dir('dunnhumby')}_history_linear_nv_es_v2",
                    reuse_dirs=base.configure_strength(m4_mode="complementary").reuse_dirs)
    return validate(Config(**(defaults | overrides)))


def strength_cfg(cfg, epochs=100):
    return base.StrengthConfig(**{k: (epochs if k == 'epochs' else getattr(cfg, k))
                                for k in base.StrengthConfig.__dataclass_fields__})


def validate(cfg):
    fixed.validate(strength_cfg(cfg))
    if (cfg.epochs, cfg.eval_every, cfg.patience_start, cfg.patience) != (300,25,100,4):
        raise ValueError("Registered early stop fixes max=300, eval=25, start=100, patience=4")
    return cfg


def specs(cfg):
    return [s for s in fixed.new_specs() if cfg.include_controls or s['condition'] == 'nv']


def anchor_specs(cfg):
    return [s for s in base.arm_specifications(strength_cfg(cfg)) if s['role'] in ('M1','M4','M2')]


def replay(curve, cfg, *, require_complete=True):
    """Causal replay: metric ties keep the earlier model; no look-ahead."""
    evaluated = [r for r in curve if 'metrics' in r]
    best = None
    waits = 0
    seen = []
    for expected, record in enumerate(evaluated, start=1):
        epoch = int(record['epoch'])
        if epoch != expected * cfg.eval_every or epoch > cfg.epochs:
            raise ValueError('Evaluation grid is missing, duplicated or out of order')
        score = float(record['metrics'][MONITOR])
        if not np.isfinite(score):
            raise ValueError('Selection metric must be finite')
        improved = best is None or score > float(best['metrics'][MONITOR])
        if improved:
            best = record
        if epoch <= cfg.patience_start:
            waits = 0
        else:
            waits = 0 if improved else waits+1
        seen.append(record)
        complete = waits >= cfg.patience or epoch == cfg.epochs
        if complete:
            return dict(selected_epoch=best['epoch'], stopped_epoch=epoch,
                        selected_record=best, waits=waits, complete=True,
                        stop_reason='patience' if waits >= cfg.patience else 'max_epochs', curve=seen)
    if require_complete:
        raise ValueError('Early-stop trajectory incomplete')
    return dict(selected_epoch=best['epoch'] if best else None,
                selected_record=best, stopped_epoch=seen[-1]['epoch'] if seen else 0,
                waits=waits, complete=False, stop_reason=None, curve=seen)


def preflight(cfg):
    p = fixed.preflight(strength_cfg(cfg))
    p.update(code_version=CODE_VERSION, config=asdict(cfg), epochs=cfg.epochs,
             new_fits=len(cfg.seeds)*len(specs(cfg)),
             selection=dict(monitor=MONITOR, mode='max', eval_every=25, min_delta=0,
                            patience_start=100, patience_evaluations=4,
                            ties='earlier checkpoint', restore_best=True),
             anchor_policy='replay compatible full-prefix curves; missing M1/M4 require explicit per-arm training approval; old M2 reuse-only',
             reading='development selection; no significance/generalization/CLV attribution claim',
             primary_reference='M2 vs selected M1; M5 vs selected complementary M4',
             practical_goal='also report both economic metrics vs selected M1; do not confuse M5-M4 gain with overall goal')
    return p


def _identity(prepared, cfg, spec, seed):
    identity = base._identity(prepared, strength_cfg(cfg, cfg.epochs), spec, seed)
    identity['version'] = CODE_VERSION
    identity['selection'] = preflight(cfg)['selection']
    for name in (Path(__file__).name, 'clv_history_linear_nv_model.py', 'lightgcn_clv_history_linear_nv.py'):
        identity['source_hashes'][name] = file_sha256(Path(__file__).with_name(name))
    return identity


def _paths(prepared, cfg, spec, seed):
    identity = _identity(prepared, cfg, spec, seed)
    root = Path(cfg.out_dir) / 'arms' / base._digest(identity)
    store = ProgressStore(root / 'progress', RunIdentity(CODE_VERSION, spec['model_id'], seed,
        base._digest(identity), 'content:'+base._digest(identity['source_hashes']), prepared['input_hash']))
    return identity, root, store


def _cached(prepared, cfg, spec, seed):
    identity, root, _ = _paths(prepared,cfg,spec,seed)
    path = root / 'result.json'
    if not path.is_file():
        return None
    p = json.loads(path.read_text())
    if p.get('identity') != identity:
        raise RuntimeError('Early-stop cache identity mismatch')
    replayed = replay(p['curve'], cfg)
    if (p['selected_epoch'] != replayed['selected_epoch']
            or base._digest(p['metrics']) != base._digest(replayed['selected_record']['metrics'])):
        raise RuntimeError('Selected metrics do not match chronological replay')
    return p


def reuse_curve(prepared, cfg, spec, seed):
    """Only known capacity-run provenance; never CSV guesses or lone best rows."""
    if spec['role'] not in ('M1','M2'):
        return None
    old_id = base.capacity.M1_MODEL_ID if spec['role']=='M1' else base.capacity.M2_MODEL_ID
    matches = []
    for directory in cfg.reuse_dirs:
        for summary_path in sorted(Path(directory).glob('clv_m2_capacity_search_*.json')):
            summary = json.loads(summary_path.read_text())
            if summary.get('code_version') != base.capacity.CODE_VERSION:
                continue
            stored = summary.get('config', {})
            if seed not in stored.get('seeds', []) or 'baseline' not in stored.get('conditions', []):
                continue
            if any(stored.get(k) != getattr(cfg,k) for k in ('epochs','eval_every','batch_size','lr','n_layers','negative_count')):
                continue
            old_cfg = base.capacity.CapacitySearchConfig(**stored)
            key = base.capacity._config_hash(old_cfg, prepared['input_hash'])
            if summary_path.stem != f'clv_m2_capacity_search_{key}':
                continue
            path = Path(directory) / 'arms' / key / f'baseline_{old_id}_s{seed}.json'
            if not path.is_file():
                continue
            p = json.loads(path.read_text())
            expected = dict(code_version=base.capacity.CODE_VERSION, model_id=old_id, condition='baseline',
                seed=seed, id_dim=cfg.id_dim, pref_reg=cfg.pref_reg, rho=spec['rho'],
                axis_dim=0 if spec['role']=='M1' else cfg.history_axis_dim)
            if any(p.get(k)!=v for k,v in expected.items()):
                continue
            selected = replay(p['curve'], cfg)
            metrics = selected['selected_record']['metrics']
            if not all(np.isfinite(metrics.get(k,np.nan)) for k in (*base.ACCURACY,*fixed.PRIMARY)):
                raise RuntimeError(f'Incomplete baseline metrics: {path}')
            matches.append(dict(**spec, seed=seed, origin='replayed_capacity_curve',
                metrics=metrics, selected_epoch=selected['selected_epoch'],
                stopped_epoch=selected['stopped_epoch'], stop_reason=selected['stop_reason'],
                curve=selected['curve'], diagnostics=selected['selected_record'].get('score_split', {}),
                source_result=str(path), source_sha256=file_sha256(path),
                checkpoint=None, checkpoint_note='readout reuse; selected weights may not be saved',
                original_run_epochs=stored['epochs']))
    if len({base._digest((p['curve'],p['metrics'])) for p in matches})>1:
        raise RuntimeError(f'Conflicting compatible curves: {seed} {old_id}')
    return matches[0] if matches else None


def prepare(cfg):
    validate(cfg)
    prepared = base._prepare(strength_cfg(cfg))
    anchors, plan = [], []
    for seed in cfg.seeds:
        for spec in anchor_specs(cfg) + specs(cfg):
            result = _cached(prepared,cfg,spec,seed)
            if result is None and spec['role'] in ('M1','M2') and 'linear' not in spec['model_id']:
                result = reuse_curve(prepared,cfg,spec,seed)
            if result is not None:
                anchors.append(result)
                action, remaining = 'reuse_completed', 0
            elif spec['model_id'] == 'm2_rho0.05':
                action, remaining = 'optional_old_M2_unavailable_no_training', 0
            else:
                _, _, store = _paths(prepared,cfg,spec,seed)
                if store.latest_checkpoint.is_file():
                    state = torch.load(store.latest_checkpoint, map_location='cpu', weights_only=False)
                    store._validate_identity(state['identity'])
                    action, remaining = 'resume', max(0,cfg.epochs-int(state['epoch']))
                else:
                    action, remaining = 'train_from_start', cfg.epochs
            plan.append(dict(seed=seed, model_id=spec['model_id'], action=action,
                             max_additional_epochs=remaining))
    prepared.update(es_config=asdict(cfg), anchors=anchors, plan=plan)
    print(pd.DataFrame(plan).to_string(index=False), flush=True)
    print('No training performed. Missing M1/M4 need explicit per-arm approval.',flush=True)
    return prepared


def _train(model, prepared, cfg, spec, seed, store, root):
    data = prepared['data']
    tr_u, tr_i = data['tr_u'],data['tr_i']
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    # Fail closed rather than deleting a corrupt resume and silently restarting.
    if store.latest_checkpoint.is_file():
        torch.load(store.latest_checkpoint, map_location='cpu', weights_only=False)
    state = store.restore_epoch(model,optimizer,rng)
    history = list(state.get('history',[])) if state else []
    start = int(state['next_epoch']) if state else 1
    prior_wall = float(state.get('wall_clock_sec',0)) if state else 0.
    selection = replay(history,cfg,require_complete=False)
    started = time.time()
    weights = torch.as_tensor(prepared['m4_weights'],device=base.v3.DEVICE,dtype=torch.float32) if spec['weighted'] else None
    stopped = start-1
    if not selection['complete']:
        for epoch in range(start,cfg.epochs+1):
            model.train()
            epoch_start = time.time()
            indices = rng.permutation(len(tr_u))
            totals = np.zeros(3)
            batches = math.ceil(len(tr_u)/cfg.batch_size)
            for batch in range(batches):
                ix = indices[batch*cfg.batch_size:(batch+1)*cfg.batch_size]
                negatives = base.components.m4_helpers.sample_uniform_negative_matrix(
                    tr_u[ix],tr_i[ix],data['n_items'],data['pos_key'],rng,k=1)
                u,p,n = [torch.as_tensor(a,dtype=torch.long,device=base.v3.DEVICE)
                         for a in (tr_u[ix],tr_i[ix],negatives)]
                w = None if weights is None else weights[torch.as_tensor(ix,device=base.v3.DEVICE)]
                loss,bpr,correct = base.components._batch_loss(model,u,p,n,w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals += (float(loss.detach()),bpr,correct)
                store.heartbeat(epoch=epoch,max_epoch=cfg.epochs,batch=batch+1,batches=batches,
                                selection=MONITOR)
            record = dict(epoch=epoch,loss=totals[0]/batches,bpr=totals[1]/batches,
                          p_correct=totals[2]/batches,epoch_sec=time.time()-epoch_start)
            if epoch % cfg.eval_every == 0:
                record['metrics'] = base.capacity._evaluate(model,prepared)
                record['diagnostics'] = {**model.representation_diagnostics(),
                    **model.training_gradient_diagnostics(), **base.hm_budget._score_share(model,prepared)}
            history.append(record)
            selection = replay(history,cfg,require_complete=False)
            if selection['selected_epoch'] == epoch:
                _atomic_torch(root/f'selected_epoch{epoch}.pt',dict(model_state=clone_state(model),epoch=epoch))
            store.save_epoch(model,optimizer,rng,epoch=epoch,history=history,
                wall_clock_sec=prior_wall+time.time()-started,selection=MONITOR,
                best_epoch=selection['selected_epoch'] or 0,waits=selection['waits'])
            stopped=epoch
            print(f"[{spec['model_id']} s{seed}] ep {epoch}/{cfg.epochs} | loss {record['loss']:.4f} | {record['epoch_sec']:.0f}s"
                  + (f" | {MONITOR}={record['metrics'][MONITOR]:.6f} | wait {selection['waits']}/{cfg.patience}" if 'metrics' in record else ''),flush=True)
            if selection['complete']:
                break
    selection = replay(history,cfg)
    checkpoint = root/f"selected_epoch{selection['selected_epoch']}.pt"
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'])
    return dict(selected_epoch=selection['selected_epoch'],stopped_epoch=stopped,
        stop_reason=selection['stop_reason'],metrics=selection['selected_record']['metrics'],
        diagnostics=selection['selected_record'].get('diagnostics',{}),curve=selection['curve'],
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        training=dict(history=history,resumed_from_epoch=start-1,wall_clock_sec=prior_wall+time.time()-started))


def _run_arm(prepared,cfg,spec,seed):
    cached = _cached(prepared,cfg,spec,seed)
    if cached is not None:
        return cached
    identity,root,store = _paths(prepared,cfg,spec,seed)
    model = (fixed._build(prepared,strength_cfg(cfg),spec,seed) if 'linear' in spec['model_id']
             else base._build_model(prepared,strength_cfg(cfg),spec,seed))
    result = _train(model,prepared,cfg,spec,seed,store,root)
    payload = dict(**spec,seed=seed,identity=identity,origin='trained_early_stop',**result,
                   data_diagnostics=prepared.get('data_diagnostics',{}),
                   m4_diagnostics=prepared.get('m4_diagnostics',{}) if spec['weighted'] else {})
    base.capacity.test10._atomic_json(root/'result.json',payload)
    store.mark_complete(epoch=result['stopped_epoch'],max_epoch=cfg.epochs,
        best_epoch=result['selected_epoch'],selection=MONITOR,result_path=str(root/'result.json'),
        checkpoint_path=result['checkpoint'])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def save(arms,cfg):
    absolute,comparison,summary,reading = fixed.tables(arms,cfg.seeds)
    reading=reading[reading.model_id.isin([s['model_id'] for s in specs(cfg)])].copy()
    selected = pd.DataFrame([dict(seed=a['seed'],model_id=a['model_id'],selected_epoch=a['selected_epoch'],
                                stopped_epoch=a['stopped_epoch'],stop_reason=a['stop_reason']) for a in arms])
    absolute=absolute.merge(selected,on=['seed','model_id'],validate='one_to_one')
    # Business goal vs M1 is distinct from the incremental M5-M4 result.
    reading['economic_gain_vs_m1']=False
    for idx,row in reading.iterrows():
        wanted=set(cfg.seeds) if row.seed=='mean' else {row.seed}
        ok=True
        for metric in fixed.PRIMARY:
            p=comparison[(comparison.model_id==row.model_id)&(comparison.reference=='m1')&
                         (comparison.metric==metric)&comparison.seed.isin(wanted)]
            ok &= set(p.seed)==wanted and np.isfinite(p[['value','reference_value']].to_numpy()).all() and p.delta.mean()>0
        reading.loc[idx,'economic_gain_vs_m1']=bool(ok)
    reading['overall_goal_direction']=reading.directional_pass & reading.economic_gain_vs_m1
    curves=pd.DataFrame([dict(seed=a['seed'],model_id=a['model_id'],epoch=r['epoch'],**r['metrics'])
                         for a in arms for r in a['curve']])
    root=Path(cfg.out_dir)/'reports'/base._digest(dict(config=asdict(cfg),version=CODE_VERSION))
    paths={}
    for name,frame in [('absolute',absolute),('comparison',comparison),('summary',summary),('reading',reading),('curve',curves)]:
        path=root/f'{name}.csv'
        base.capacity.test10._atomic_csv(path,frame)
        paths[name]=str(path)
    paths['json']=str(root/'result.json')
    base.capacity.test10._atomic_json(root/'result.json',dict(preflight=preflight(cfg),arms=arms,
        reading=reading.to_dict('records'),paths=paths))
    absolute.attrs['paths']=paths
    return absolute


def run(cfg,prepared=None,*,approved_baseline_fits=()):
    validate(cfg)
    prepared = prepare(cfg) if prepared is None else prepared
    if prepared.get('es_config')!=asdict(cfg):
        raise ValueError('Configuration changed after prepare')
    missing={f"{p['seed']}:{p['model_id']}" for p in prepared['plan']
             if p['model_id'] in ('m1','m4') and p['action']=='train_from_start'}
    if missing-set(approved_baseline_fits):
        raise RuntimeError('Additional baseline fits require explicit approval before ANY training: '+', '.join(sorted(missing)))
    arms=list(prepared['anchors'])
    for seed in cfg.seeds:
        required=[s for s in anchor_specs(cfg) if s['role'] in ('M1','M4')]+specs(cfg)
        for spec in required:
            if any(a['seed']==seed and a['model_id']==spec['model_id'] for a in arms):
                continue
            arms.append(_run_arm(prepared,cfg,spec,seed))
            if any(a['model_id']=='m1' for a in arms) and len(arms)>1:
                save(arms,cfg)
    return save(arms,cfg)
