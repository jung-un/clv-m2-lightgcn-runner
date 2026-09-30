"""M2: user N/V plus item price ranks within coarse and fine product types."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from clv_shared_feature_residual_m2 import (
    AXES, SharedFeatureResidualLightGCN, build_features, centered_basis,
)
from lightgcn_clv_v3 import SCHEMA

FINE_AXIS = 'item_fine_within'


def build_fine_features(train, *, dataset, n_users, n_items, q_n, q_v,
                        n_valid, v_valid, bandwidth=.25, meta_path=None):
    """All price ranks use training rows; metadata supplies only static taxonomy."""
    if dataset not in ('dunnhumby', 'hm') or 'i_raw' not in train:
        raise ValueError('Known dataset and training item keys required')
    features = build_features(train, n_users=n_users, n_items=n_items, q_n=q_n,
        q_v=q_v, n_valid=n_valid, v_valid=v_valid, bandwidth=bandwidth)
    schema = SCHEMA[dataset]
    fine_col = 'SUB_COMMODITY_DESC' if dataset == 'dunnhumby' else 'product_type_name'
    coarse_col = schema['category_col']
    meta = pd.read_csv(meta_path or schema['item_meta_path'],
        usecols=[schema['item_key_col'], coarse_col, fine_col],
        dtype={schema['item_key_col']: str})
    meta = meta.rename(columns={schema['item_key_col']: 'key'})
    meta['key'] = meta['key'].str.zfill(10) if dataset == 'hm' else meta['key']
    if meta.key.isna().any() or meta.key.duplicated().any():
        raise ValueError('Metadata item keys must be unique and nonmissing')
    item = train[['i_idx', 'i_raw', 'up']].copy()
    item['key'] = item.i_raw.astype(str)
    item['key'] = item.key.str.zfill(10) if dataset == 'hm' else item.key
    item['up'] = item.up.where(np.isfinite(item.up) & (item.up > 0))
    item = item.groupby(['i_idx', 'key'], as_index=False).up.median()
    item = item.merge(meta, on='key', how='left', validate='one_to_one')
    item[fine_col] = item[fine_col].replace('', np.nan)
    item[coarse_col] = item[coarse_col].replace('', np.nan)
    valid = item.up.notna() & item[fine_col].notna() & item[coarse_col].notna()
    item['rank'] = np.nan
    group_cols = [coarse_col, fine_col]
    support = item.loc[valid].groupby(group_cols, dropna=False).up
    counts = support.transform('size')
    distinct = support.transform('nunique')
    eligible = valid.copy()
    eligible.loc[valid] = (counts >= 2).to_numpy() & (distinct >= 2).to_numpy()
    item.loc[eligible, 'rank'] = item.loc[eligible].groupby(group_cols).up.rank(pct=True)
    values = np.full(n_items, .5, np.float64)
    mask = np.zeros(n_items, bool)
    idx = item.loc[eligible, 'i_idx'].to_numpy(np.int64)
    values[idx] = item.loc[eligible, 'rank'].to_numpy(np.float64)
    mask[idx] = True
    features[FINE_AXIS], info = centered_basis(values, mask, bandwidth)
    info.update(metadata_column=fine_col, static_metadata=True,
        valid_item_count=int(mask.sum()), missing_or_singleton_items=int(n_items-mask.sum()),
        valid_type_count=int(item.loc[eligible, group_cols].drop_duplicates().shape[0]),
        rank_definition='training median unit price percentile within coarse+fine type; singleton/constant/missing types masked')
    features['diagnostics'][FINE_AXIS] = info
    return features


class FinePriceLightGCN(SharedFeatureResidualLightGCN):
    """The fifth economic feature enters layer 0 and the same BPR optimizer."""

    def __init__(self, *, features, axis_l2=None, **kwargs):
        axis_l2 = dict.fromkeys((*AXES, FINE_AXIS), .001) if axis_l2 is None else dict(axis_l2)
        if set(axis_l2) != set((*AXES, FINE_AXIS)) or not np.isfinite(axis_l2[FINE_AXIS]) or axis_l2[FINE_AXIS] <= 0:
            raise ValueError('Positive finite L2 required for all five feature axes')
        super().__init__(features=features, axis_l2={a: axis_l2[a] for a in AXES}, **kwargs)
        x = np.asarray(features[FINE_AXIS], np.float32)
        if x.shape != (self.n_items, 3) or not np.isfinite(x).all():
            raise ValueError('Invalid fine-type economic input')
        self.fine_encoder = nn.Linear(3, self.id_dim, bias=False)
        nn.init.zeros_(self.fine_encoder.weight)
        self.register_buffer('fine_input', torch.from_numpy(x.copy()), persistent=False)
        self.fine_l2 = float(axis_l2[FINE_AXIS])

    def embeddings(self, need_value=True):
        user, item = self.E_u.weight, self.E_i.weight
        for name, layer in self.encoders.items():
            if name in self.disabled_axes:
                continue
            value = layer(getattr(self, name+'_input'))
            if name.startswith('user'):
                user = user+value
            else:
                item = item+value
        if FINE_AXIS not in self.disabled_axes:
            item = item+self.fine_encoder(self.fine_input)
        current = total = torch.cat([user, item])
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total+current
        total = total/(self.n_layers+1)
        return (total[:self.n_users], total[self.n_users:],
                total.new_zeros((self.n_users, 1)), total.new_zeros((self.n_items, 1)))

    def batch_l2(self, users, positives, negatives):
        return super().batch_l2(users, positives, negatives)+self.fine_l2*self.fine_encoder.weight.square().sum()

    @torch.no_grad()
    def representation_diagnostics(self):
        result = super().representation_diagnostics()
        output = self.fine_encoder(self.fine_input)
        result['shared_parameters'] += self.fine_encoder.weight.numel()
        result[FINE_AXIS+'_mean_norm'] = float(output.norm(dim=1).mean())
        result[FINE_AXIS+'_coordinate_std'] = float(output.std(dim=0, unbiased=False).mean())
        result[FINE_AXIS+'_l2_penalty'] = float(self.fine_l2*self.fine_encoder.weight.square().sum())
        return result
