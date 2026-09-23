"""One-factor M5 screen and train-only M4 weight audit; old runners unchanged."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_history_linear_nv_early_stop as es
from clv_history_item_fit_model import HistoryItemFitLightGCN
from clv_history_linear_nv_model import LinearNVHistoryLightGCN
from clv_run_state import ProgressStore, RunIdentity, file_sha256

VERSION = 'linear-nv-affine-l2-seed43-development-v1'
AFFINE_REG = 1e-4
MODEL_ID = 'm5_linear_nv_affine_l2_1e-4'
EXPECTED_REPORT_SHA = '57d46918442ecfd6d8c04f99b6d1d2f5c0f4df016d36b9da03590565208af3cb'
base = es.base
# Reviewed diff d3b5233..45e625b: seed45 allowlist and arm filtering only.
COMPATIBLE_SOURCE_PAIRS = {
    'lightgcn_clv_history_m5_strength.py': (
        'd4d2a3d2f5a1d3ee06293f6708c7a40b5695f1a03976a9f647083f96953943b2',
        'a55567324c9e62a8c0d0bdb5cfb7cc4f1724705183e3b190e461f25e0aa16c7c'),
    'lightgcn_clv_history_linear_nv_early_stop.py': (
        'f3515b94e915eedef5515ac55da674e45e851345ca1bb2a4942bbbb48687c198',
        'af3af46714472aa7d9773198d23ebe501bfff1725edf68aaa196fd4f3cfb6309'),
}


class AffineL2HistoryLightGCN(LinearNVHistoryLightGCN):
    """Identical forward/initialization; only shared affine L2 changes."""
    def batch_l2(self, users, positives, negatives, need_value=False):
        shared = sum(p.square().sum() for layer in (self.n_transform, self.v_transform)
                     for p in layer.parameters())
        return HistoryItemFitLightGCN.batch_l2(self, users, positives, negatives, need_value) + AFFINE_REG*shared

    def representation_diagnostics(self):
        return dict(super().representation_diagnostics(), affine_reg=AFFINE_REG,
                    embedding_reg=self.pref_reg)


def configure(out_dir):
    return es.configure(seeds=(43,), include_m2=False, include_controls=False, out_dir=str(out_dir))


def _spec():
    return dict(es.specs(configure('unused'))[0], model_id=MODEL_ID, affine_reg=AFFINE_REG)


def build(prep, cfg):
    base.v3.set_seed(43)
    d = prep['data']; valid = np.asarray(prep['clv_valid'], bool)
    return AffineL2HistoryLightGCN(n_users=d['n_users'], n_items=d['n_items'], history=prep['history'],
        q_n=np.where(valid, prep['q_n'], 0), q_v=np.where(valid, prep['q_v'], 0),
        activity_valid=valid, value_valid=valid, adj=d['adj'], id_dim=cfg.id_dim,
        axis_dim=cfg.history_axis_dim, n_layers=cfg.n_layers, rho=.05,
        pref_reg=cfg.pref_reg, constant_q=False).to(base.v3.DEVICE)


def verified_anchors(report_path, cfg, prep):
    """Exact user-supplied seed43 report, no search/guess/retraining fallback."""
    path = Path(report_path)
    if not path.is_file() or file_sha256(path) != EXPECTED_REPORT_SHA:
        raise ValueError('Expected seed43 result.json missing or changed; no training permitted')
    report = json.loads(path.read_text()); pf = report['preflight']
    expected = json.loads(json.dumps(es.preflight(cfg)))
    for key in ('code_version', 'selection', 'split', 'final_test', 'holdout'):
        if pf.get(key) != expected[key]:
            raise ValueError(f'Baseline protocol mismatch: {key}')
    for key, value in expected['config'].items():
        if key not in ('out_dir', 'reuse_dirs', 'include_m2') and pf['config'].get(key) != value:
            raise ValueError(f'Baseline config mismatch: {key}')
    anchors = []
    for mid in ('m1', 'm4', 'm5_linear_nv'):
        matches = [a for a in report['arms'] if a['model_id'] == mid and a['seed'] == 43]
        if len(matches) != 1:
            raise ValueError(f'Exactly one seed43 {mid} required')
        a = matches[0]; replay = es.replay(a['curve'], cfg)
        if (a['selected_epoch'] != replay['selected_epoch'] or a['stopped_epoch'] != replay['stopped_epoch']
                or a['metrics'] != replay['selected_record']['metrics']):
            raise ValueError('Stored baseline selection differs from chronological replay')
        if mid != 'm1':
            identity = a['identity']
            if identity['input_hash'] != prep['input_hash']:
                raise ValueError('Input hash differs from baseline')
            for name, digest in identity['source_hashes'].items():
                current = file_sha256(Path(__file__).with_name(name))
                if current != digest and COMPATIBLE_SOURCE_PAIRS.get(name) != (digest, current):
                    raise ValueError(f'Baseline source changed: {name}')
        else:
            # Capacity-curve provenance is validated by the existing strict loader.
            old = es.reuse_curve(prep, cfg, es.anchor_specs(cfg)[0], 43)
            if old is None or old['metrics'] != a['metrics']:
                raise ValueError('M1 capacity provenance unavailable/mismatched; no fallback fit')
        if not all(np.isfinite(a['metrics'].get(k, np.nan)) for k in (*base.ACCURACY, *es.fixed.PRIMARY)):
            raise ValueError('Required baseline metrics missing')
        anchors.append(dict(a, origin='reused_exact_seed43_report'))
    return anchors


def weight_audit(prep, cfg):
    """Exact training rows, not users or realized optimizer/loss contributions."""
    d=prep['data']; u=np.asarray(d['tr_u']); i=np.asarray(d['tr_i'])
    fit=np.clip(base.weights_module.value_basis_fit(prep,cfg),0,1)
    q=np.asarray(prep['q_c'])[u]; price=np.asarray(prep['item_amount_percentile'])[i]
    valid=np.asarray(prep['clv_valid'],bool)[u] & np.asarray(prep['item_economic_valid'],bool)[i]
    raw=1+.5*q*(1-fit); w=raw/raw.mean()
    if not np.allclose(w,prep['m4_weights'],rtol=1e-6,atol=1e-8):
        raise ValueError('Audited weights do not match actual training weights')
    clv=np.asarray(prep['axes']['clv_proxy'])
    edges=base.v3.segment_thresholds(clv,prep['base_cfg']['SEG_EDGES'])
    # Match evaluation segmentation including equal-threshold behavior.
    lo,hi=edges
    labels=np.where(clv[u]<=lo,'저CLV',np.where(clv[u]>=hi,'고CLV','중CLV'))
    def bins(values):
        return pd.cut(values,[-np.inf,.25,.5,.75,np.inf],labels=['<=.25','(.25,.50]','(.50,.75]','>.75']).astype(str)
    direction=np.where(price>np.asarray(prep['q_v'])[u],'item_percentile_above_qV',
                       np.where(price<np.asarray(prep['q_v'])[u],'item_percentile_below_qV','equal'))
    frame=pd.DataFrame(dict(segment=labels, price_bin=bins(price), fit_bin=bins(fit),
        validity=np.where(valid,'valid','invalid'), percentile_direction=direction,
        weight=w, extra=raw-1, price=price, fit=fit, q_c=q))
    outputs=[]
    for keys in (['segment'],['price_bin'],['fit_bin'],['percentile_direction'],['validity'],
                 ['segment','price_bin','fit_bin'],['segment','percentile_direction']):
        part=frame.groupby(keys,dropna=False,observed=True).agg(
            n_rows=('weight','size'),weight_sum=('weight','sum'),mean_weight=('weight','mean'),
            extra_sum=('extra','sum'),mean_price_percentile=('price','mean'),
            mean_fit=('fit','mean'),mean_q_c=('q_c','mean')).reset_index()
        part['grouping']=' × '.join(keys)
        part['row_share']=part.n_rows/len(frame)
        part['weight_share']=part.weight_sum/w.sum()
        part['weight_share_over_row_share']=part.weight_share/part.row_share
        part['extra_share']=part.extra_sum/frame.extra.sum() if frame.extra.sum()>0 else np.nan
        outputs.append(part)
    return pd.concat(outputs,ignore_index=True), dict(n_train_rows=len(frame),mean_raw=float(raw.mean()),
        weight_cv=float(w.std()/w.mean()),segment_thresholds=list(map(float,edges)),
        no_training=True,scope='train rows; weight mass is NOT loss mass or performance attribution',
        percentile_direction_note='item amount percentile vs q_V; not a raw-price difference')


def prepare(report_path, out_dir):
    cfg=configure(out_dir)
    # Check source first, before expensive data preparation.
    if not Path(report_path).is_file() or file_sha256(report_path)!=EXPECTED_REPORT_SHA:
        raise ValueError('Missing/changed seed43 report; no training started')
    prep=base._prepare(es.strength_cfg(cfg))
    anchors=verified_anchors(report_path,cfg,prep)
    audit,meta=weight_audit(prep,cfg)
    root=Path(out_dir); audit_path=root/'m4_weight_distribution.csv'
    base.capacity.test10._atomic_csv(audit_path,audit)
    meta.update(report=str(report_path),report_sha256=file_sha256(report_path),input_hash=prep['input_hash'])
    base.capacity.test10._atomic_json(root/'m4_weight_diagnostic.json',meta)
    prep.update(anchors=anchors,screen_config=asdict(cfg),source_report=str(report_path),
                source_report_sha256=file_sha256(report_path))
    print('M4 진단 완료. M1/M4/기존 M5 재사용. 다음 학습 셀은 변경 M5 한 개만 실행합니다.',flush=True)
    return cfg,prep,audit


def _identity(prep,cfg):
    identity=es._identity(prep,cfg,_spec(),43)
    identity.update(version=VERSION,affine_reg=AFFINE_REG,baseline_report_sha256=prep['source_report_sha256'])
    identity['source_hashes'][Path(__file__).name]=file_sha256(__file__)
    return identity


def save(arms,cfg,prep):
    absolute,comparison,_,_=es.fixed.tables(arms,(43,))
    candidate=next(a for a in arms if a['model_id']==MODEL_ID)
    old=next(a for a in arms if a['model_id']=='m5_linear_nv')
    extra=[]
    for metric,value in candidate['metrics'].items():
        ref=old['metrics'].get(metric)
        if isinstance(value,(float,int)) and isinstance(ref,(float,int)):
            extra.append(dict(seed=43,model_id=MODEL_ID,reference='m5_linear_nv',metric=metric,
                value=value,reference_value=ref,delta=value-ref,relative_change_pct=100*(value/ref-1) if ref else np.nan))
    comparison=pd.concat([comparison,pd.DataFrame(extra)],ignore_index=True)
    selected=pd.DataFrame([dict(model_id=a['model_id'],selected_epoch=a['selected_epoch'],
        stopped_epoch=a['stopped_epoch']) for a in arms])
    absolute=absolute.merge(selected,on='model_id',validate='one_to_one')
    metrics={a['model_id']:a['metrics'] for a in arms}; new=metrics[MODEL_ID]
    reading={'accuracy_guard':all(new[k]>=.99*metrics['m1'][k] for k in base.ACCURACY)}
    for ref in ('m1','m4','m5_linear_nv'):
        reading[f'both_economic_at10_above_{ref}']=all(new[k]>metrics[ref][k] for k in es.fixed.PRIMARY)
    reading.update(significance_claim=False,scope='single development seed; report @20/@50 and every segment too')
    curve=pd.DataFrame([dict(model_id=a['model_id'],epoch=r['epoch'],**r['metrics']) for a in arms for r in a['curve']])
    root=Path(cfg.out_dir)/'reports';paths={}
    for name,df in [('absolute',absolute),('comparison',comparison),('curve',curve)]:
        path=root/f'{name}.csv';base.capacity.test10._atomic_csv(path,df);paths[name]=str(path)
    paths['json']=str(root/'result.json')
    base.capacity.test10._atomic_json(Path(paths['json']),dict(code_version=VERSION,config=asdict(cfg),
        affine_reg=AFFINE_REG,selection=es.preflight(cfg)['selection'],reading=reading,arms=arms,
        final_test=False,holdout=False,source_report=prep['source_report'],paths=paths))
    return paths


def run(cfg,prep):
    if asdict(cfg)!=prep['screen_config'] or cfg!=configure(cfg.out_dir):
        raise ValueError('Configuration changed after prepare')
    # Revalidate report and provenance before any training.
    anchors=verified_anchors(prep['source_report'],cfg,prep)
    identity=_identity(prep,cfg);key=base._digest(identity);root=Path(cfg.out_dir)/'arms'/key
    result_path=root/'result.json'
    if result_path.is_file():
        candidate=json.loads(result_path.read_text())
        if candidate['identity']!=identity:
            raise ValueError('Candidate cache identity mismatch')
        selected=es.replay(candidate['curve'],cfg)
        if candidate['metrics']!=selected['selected_record']['metrics']:
            raise ValueError('Candidate cache selection mismatch')
    else:
        store=ProgressStore(root/'progress',RunIdentity(VERSION,MODEL_ID,43,key,
            'content:'+base._digest(identity['source_hashes']),prep['input_hash']))
        model=build(prep,cfg)
        result=es._train(model,prep,cfg,_spec(),43,store,root)
        candidate=dict(**_spec(),seed=43,origin='trained_affine_l2_screen',identity=identity,**result,
                       data_diagnostics=prep.get('data_diagnostics',{}),m4_diagnostics=prep['m4_diagnostics'])
        base.capacity.test10._atomic_json(result_path,candidate)
        store.mark_complete(epoch=result['stopped_epoch'],max_epoch=cfg.epochs,
            best_epoch=result['selected_epoch'],result_path=str(result_path),checkpoint_path=result['checkpoint'])
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return save(anchors+[candidate],cfg,prep)
