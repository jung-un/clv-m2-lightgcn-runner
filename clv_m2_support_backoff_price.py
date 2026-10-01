"""M2: user N/V and one support-adaptive item-price coordinate.

The item coordinate uses fine-type price rank when available, otherwise the
coarse-category rank and finally the global rank.  Distinct training buyers
shrink the selected rank toward the neutral midpoint; no evaluation rows are
used.  All three encoders enter layer 0 and share the plain BPR optimizer.
"""

import numpy as np
import pandas as pd

from clv_shared_feature_residual_m2 import (
    SharedFeatureResidualLightGCN,
    build_features,
    centered_basis,
)
from clv_m5_n_conditioned_value_basis_model import fixed_value_basis
from lightgcn_clv_v3 import item_value_features

PRICE_AXIS = 'item_support_backoff_price'
AXES = ('user_n', 'user_v', PRICE_AXIS)


def reliability_gated_centered_basis(values, valid, reliability, bandwidth=.25):
    """Center the price basis, then make zero support exactly non-informative."""
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    reliability = np.asarray(reliability, float)
    if values.ndim != 1 or valid.shape != values.shape or reliability.shape != values.shape:
        raise ValueError('Price values, validity and reliability must be aligned vectors')
    if (not np.isfinite(reliability).all()
            or np.any((reliability < 0) | (reliability > 1))):
        raise ValueError('Reliability must be finite and in [0,1]')
    if valid.any() and (not np.isfinite(values[valid]).all()
                        or np.any((values[valid] < 0) | (values[valid] > 1))):
        raise ValueError('Valid price percentiles must be finite and in [0,1]')
    basis = fixed_value_basis(np.where(valid, values, .5), valid, bandwidth=bandwidth)
    weight = np.where(valid, reliability, 0.)
    center = np.average(basis, axis=0, weights=weight) if weight.sum() else np.zeros(3)
    result = (weight[:, None] * (basis - center)).astype(np.float32)
    result[~valid] = 0.
    return result, dict(
        valid_count=int(valid.sum()),
        unique_valid=int(np.unique(values[valid]).size),
        center=center.tolist(),
        centered_rms=float(np.sqrt(np.mean(result**2))),
        zero_reliability_count=int(np.sum(valid & (reliability == 0))),
        reliability_gate_after_basis=True,
    )


def support_backoff_price(*, global_price, coarse_price, fine_price, fine_valid,
                          coarse_valid, observation_count, shrinkage=10.0):
    """Return one train-only price rank, its validity, reliability and source."""
    arrays = [np.asarray(x) for x in (global_price, coarse_price, fine_price,
              fine_valid, coarse_valid, observation_count)]
    if any(x.ndim != 1 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError('Price coordinates, masks and support must be aligned vectors')
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError('Shrinkage must be positive and finite')
    global_price, coarse_price, fine_price = [x.astype(float) for x in arrays[:3]]
    fine_valid, coarse_valid = arrays[3].astype(bool), arrays[4].astype(bool)
    count = arrays[5].astype(float)
    if not np.isfinite(count).all() or np.any(count < 0):
        raise ValueError('Observation support must be finite and nonnegative')
    global_valid = np.isfinite(global_price)
    coarse_valid &= np.isfinite(coarse_price)
    fine_valid &= np.isfinite(fine_price)
    source = np.where(fine_valid, 'fine', np.where(coarse_valid, 'coarse', 'global'))
    selected = np.where(fine_valid, fine_price,
                        np.where(coarse_valid, coarse_price, global_price))
    valid = (count > 0) & (fine_valid | coarse_valid | global_valid)
    reliability = count / (count + float(shrinkage))
    value = .5 + reliability * (np.where(valid, selected, .5) - .5)
    value[~valid] = .5
    return value, valid, reliability, source


def build_support_backoff_features(train, *, dataset, n_users, n_items, q_n, q_v,
                                   n_valid, v_valid, bandwidth=.25,
                                   shrinkage=10.0, meta_path=None,
                                   reliability_after_basis=False):
    """Build user N/V plus one non-duplicated item economic feature."""
    required = {'u_idx', 'i_idx', 'i_raw', 'up', 'cat_idx'}
    if dataset not in ('dunnhumby', 'hm') or train.empty or not required.issubset(train):
        raise ValueError('Known dataset and nonempty train user/item/price/category rows required')
    base = build_features(train, n_users=n_users, n_items=n_items, q_n=q_n, q_v=q_v,
                          n_valid=n_valid, v_valid=v_valid, bandwidth=bandwidth)
    price, categories = item_value_features(train[['i_idx', 'up', 'cat_idx']], n_items,
                                             report=False)
    med = train.groupby('i_idx').up.median()
    observed = np.zeros(n_items, bool)
    observed[med.index.to_numpy(np.int64)] = np.isfinite(med.to_numpy())
    from lightgcn_clv_v3 import SCHEMA
    schema = SCHEMA[dataset]
    fine_col = 'SUB_COMMODITY_DESC' if dataset == 'dunnhumby' else 'product_type_name'
    coarse_col = schema['category_col']
    meta = pd.read_csv(meta_path or schema['item_meta_path'],
                       usecols=[schema['item_key_col'], coarse_col, fine_col],
                       dtype={schema['item_key_col']: str})
    meta = meta.rename(columns={schema['item_key_col']: 'key'})
    meta['key'] = meta['key'].str.zfill(10) if dataset == 'hm' else meta['key']
    item = train[['i_idx', 'i_raw', 'up']].copy()
    item['key'] = item.i_raw.astype(str)
    item['key'] = item.key.str.zfill(10) if dataset == 'hm' else item.key
    item['up'] = item.up.where(np.isfinite(item.up) & (item.up > 0))
    item = item.groupby(['i_idx', 'key'], as_index=False).up.median()
    item = item.merge(meta, on='key', how='left', validate='one_to_one')
    item[[coarse_col, fine_col]] = item[[coarse_col, fine_col]].replace('', np.nan)
    raw_valid = item.up.notna() & item[coarse_col].notna() & item[fine_col].notna()
    grouped = item.loc[raw_valid].groupby([coarse_col, fine_col], dropna=False).up
    count = grouped.transform('size')
    distinct = grouped.transform('nunique')
    eligible = raw_valid.copy()
    eligible.loc[raw_valid] = ((count >= 2) & (distinct >= 2)).to_numpy()
    item['fine_rank'] = np.nan
    item.loc[eligible, 'fine_rank'] = item.loc[eligible].groupby(
        [coarse_col, fine_col]).up.rank(pct=True)
    fine_price = np.full(n_items, np.nan)
    fine_valid = np.zeros(n_items, bool)
    idx = item.loc[eligible, 'i_idx'].to_numpy(np.int64)
    fine_price[idx] = item.loc[eligible, 'fine_rank'].to_numpy(float)
    fine_valid[idx] = True
    buyer_count = train.groupby('i_idx').u_idx.nunique()
    support = np.zeros(n_items, float)
    support[buyer_count.index.to_numpy(np.int64)] = buyer_count.to_numpy(float)
    global_price = np.where(observed, price[:, 0], np.nan)
    coarse_price = np.where(observed & (categories >= 0), price[:, 1], np.nan)
    values, valid, reliability, source = support_backoff_price(
        global_price=global_price,
        coarse_price=coarse_price,
        fine_price=fine_price, fine_valid=fine_valid,
        coarse_valid=observed & (categories >= 0), observation_count=support,
        shrinkage=shrinkage,
    )
    if reliability_after_basis:
        selected = np.where(fine_valid, fine_price,
                            np.where(observed & (categories >= 0), coarse_price, global_price))
        feature, info = reliability_gated_centered_basis(
            selected, valid, reliability, bandwidth)
    else:
        feature, info = centered_basis(values, valid, bandwidth)
        info['reliability_gate_after_basis'] = False
    info.update(
        shrinkage=float(shrinkage), support_definition='distinct training buyers per item',
        support_median=float(np.median(support[valid])) if valid.any() else 0.0,
        reliability_mean=float(reliability[valid].mean()) if valid.any() else 0.0,
        fine_source_items=int(np.sum(valid & (source == 'fine'))),
        coarse_source_items=int(np.sum(valid & (source == 'coarse'))),
        global_source_items=int(np.sum(valid & (source == 'global'))),
        single_coordinate=True,
        fine_feature_definition='training median unit price percentile within coarse+fine type',
    )
    return dict(user_n=base['user_n'], user_v=base['user_v'],
                **{PRICE_AXIS: feature},
                diagnostics=dict(user_n=base['diagnostics']['user_n'],
                                 user_v=base['diagnostics']['user_v'],
                                 **{PRICE_AXIS: info}, bandwidth=bandwidth,
                                 source='training rows and static taxonomy only'))


class SupportBackoffPriceLightGCN(SharedFeatureResidualLightGCN):
    """Three-axis layer-0 M2 with user N/V and one item-price coordinate."""

    def __init__(self, *, features, axis_l2=None, **kwargs):
        axis_l2 = dict.fromkeys(AXES, .001) if axis_l2 is None else dict(axis_l2)
        if set(axis_l2) != set(AXES):
            raise ValueError('L2 is required for user N, user V and the single price axis')
        padded = dict(features)
        padded['item_price'] = padded.pop(PRICE_AXIS)
        padded['item_within'] = np.zeros_like(padded['item_price'])
        expanded_l2 = dict(user_n=axis_l2['user_n'], user_v=axis_l2['user_v'],
                           item_price=axis_l2[PRICE_AXIS], item_within=axis_l2[PRICE_AXIS])
        super().__init__(features=padded, axis_l2=expanded_l2, **kwargs)
        self.encoders[PRICE_AXIS] = self.encoders.pop('item_price')
        self.register_buffer(PRICE_AXIS + '_input', self.item_price_input, persistent=False)
        del self.encoders['item_within']
        self.axis_l2 = dict(axis_l2)
