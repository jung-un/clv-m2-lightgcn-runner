"""Train-only N/V features, shared encoders, and jointly trained binary LightGCN.

N means historical transaction occurrence, NOT repeat purchase of the same SKU.
Item N is a shrunk buyer q_N mean; item V is a shrunk unit-price position.
Positive training items exclude the current buyer from both item aggregates.
User q_N/q_V and the global training price reference remain fixed observations.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from clv_m5_n_conditioned_value_basis_model import fixed_value_basis


def build_features(train, *, n_users, n_items, q_n, q_v, valid,
                   shrinkage=10., bandwidth=.25):
    """Return full-catalog and leave-one-buyer-out RBF inputs (no labels used)."""
    required = {'u_idx', 'i_idx', 'up'}
    if not required.issubset(train.columns) or train.empty:
        raise ValueError('Nonempty training data with u_idx/i_idx/up required')
    if min(n_users, n_items) <= 0 or not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError('Positive catalog sizes and shrinkage required')
    q_n, q_v, valid = np.asarray(q_n, float), np.asarray(q_v, float), np.asarray(valid, bool)
    if any(x.shape != (n_users,) for x in (q_n, q_v, valid)):
        raise ValueError('User percentile/mask shape mismatch')
    for q in (q_n, q_v):
        if not np.isfinite(q[valid]).all() or np.any((q[valid] < 0) | (q[valid] > 1)):
            raise ValueError('Valid user percentiles must be finite in [0,1]')
    frame = train[['u_idx', 'i_idx', 'up']].copy()
    for column, size in (('u_idx', n_users), ('i_idx', n_items)):
        values = frame[column].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values >= size) | (values != np.floor(values))):
            raise ValueError(f'{column} outside integer catalog indices')
    frame['up'] = frame.up.where(np.isfinite(frame.up) & (frame.up > 0))
    pairs = frame.groupby(['u_idx', 'i_idx'], sort=True).up.mean().reset_index()
    u, i = pairs.u_idx.to_numpy(np.int64), pairs.i_idx.to_numpy(np.int64)
    price = pairs.up.to_numpy(float)
    price_valid = np.isfinite(price)
    if not price_valid.any():
        raise ValueError('No positive observed unit prices for V axis')
    prior_price = float(price[price_valid].mean())
    n_mass = np.where(valid[u], q_n[u], 0.)
    v_mass = np.where(price_valid, price, 0.)
    n_count = np.bincount(i, weights=valid[u], minlength=n_items)
    v_count = np.bincount(i, weights=price_valid, minlength=n_items)
    n_sum = np.bincount(i, weights=n_mass, minlength=n_items)
    v_sum = np.bincount(i, weights=v_mass, minlength=n_items)
    item_n = (n_sum + shrinkage*.5)/(n_count + shrinkage)
    item_price = (v_sum + shrinkage*prior_price)/(v_count + shrinkage)
    loo_n_count, loo_v_count = n_count[i]-valid[u], v_count[i]-price_valid
    loo_n = (n_sum[i]-n_mass + shrinkage*.5)/(loo_n_count+shrinkage)
    loo_price = (v_sum[i]-v_mass + shrinkage*prior_price)/(loo_v_count+shrinkage)
    # Frozen TRAIN reference: midrank empirical CDF, also used for LOO prices.
    reference = np.sort(item_price[v_count > 0])
    def price_percentile(values):
        return (np.searchsorted(reference, values, 'left') +
                np.searchsorted(reference, values, 'right'))/(2.*len(reference))
    def basis(values, mask):
        return fixed_value_basis(np.where(mask, values, .5), mask, bandwidth=bandwidth)
    buyer_count = np.bincount(i, minlength=n_items)
    return dict(
        user_n=basis(q_n, valid), user_v=basis(q_v, valid),
        item_n=basis(item_n, n_count > 0), item_v=basis(price_percentile(item_price), v_count > 0),
        keys=u*n_items+i,
        loo_n=basis(loo_n, loo_n_count > 0),
        loo_v=basis(price_percentile(loo_price), loo_v_count > 0),
        diagnostics=dict(n_pairs=len(u), n_items=n_items,
            singleton_item_share=float((buyer_count == 1).mean()),
            valid_user_share=float(valid.mean()),
            item_n_valid_share=float((n_count > 0).mean()),
            item_v_valid_share=float((v_count > 0).mean()),
            positive_n_loo_valid_share=float((loo_n_count > 0).mean()),
            positive_v_loo_valid_share=float((loo_v_count > 0).mean()),
            q_n_unique_valid=int(np.unique(q_n[valid]).size),
            q_v_unique_valid=int(np.unique(q_v[valid]).size),
            item_n_std=float(item_n[n_count > 0].std()) if np.any(n_count > 0) else 0.,
            global_train_price_prior=prior_price, shrinkage=shrinkage, bandwidth=bandwidth,
            missing_item_policy='axis input zero; ID path retained',
            leave_out_scope='direct buyer contributions to positive item only; user CLV and global train price reference fixed'))


class SharedNVLightGCN(nn.Module):
    """S=ID dot + alpha_N*h_N(q_N) dot f_N(x_N) + alpha_V*h_V(q_V) dot f_V(x_V)."""
    supports_clv_score_split = True

    def __init__(self, *, n_users, n_items, features, adj, id_dim=64, axis_dim=4,
                 n_layers=2, pref_reg=.001, alpha_n=.05, alpha_v=.05):
        super().__init__()
        if min(n_users, n_items, id_dim, axis_dim) <= 0 or n_layers < 0:
            raise ValueError('Invalid dimensions/layer count')
        if not all(np.isfinite(x) and x > 0 for x in (alpha_n, alpha_v)):
            raise ValueError('Both N and V must have positive finite coefficients')
        if not np.isfinite(pref_reg) or pref_reg < 0:
            raise ValueError('Invalid L2 coefficient')
        if adj.layout != torch.sparse_coo or adj.shape != (n_users+n_items,)*2:
            raise ValueError('Sparse COO adjacency shape mismatch')
        self.n_users, self.n_items, self.id_dim = n_users, n_items, id_dim
        self.axis_dim, self.n_layers, self.pref_reg = axis_dim, n_layers, pref_reg
        self.alpha_n, self.alpha_v = alpha_n, alpha_v
        self.E_u, self.E_i = nn.Embedding(n_users, id_dim), nn.Embedding(n_items, id_dim)
        # Match the reused M1/M4's ID initialization order, before new parameters.
        nn.init.normal_(self.E_u.weight, std=.1)
        nn.init.normal_(self.E_i.weight, std=.1)
        self.encoders = nn.ModuleDict({name: nn.Linear(3, axis_dim, bias=False)
                                      for name in ('user_n', 'item_n', 'user_v', 'item_v')})
        for layer in self.encoders.values():
            nn.init.normal_(layer.weight, std=.1)
        self.register_buffer('adj', adj.coalesce(), persistent=False)
        keys = np.asarray(features['keys'], np.int64)
        if len(keys) == 0 or np.any(np.diff(keys) <= 0) or keys[0] < 0 or keys[-1] >= n_users*n_items:
            raise ValueError('Sorted unique training pair keys required')
        self.register_buffer('pair_keys', torch.from_numpy(keys), persistent=False)
        for name in ('user_n', 'item_n', 'user_v', 'item_v', 'loo_n', 'loo_v'):
            values = np.asarray(features[name], np.float32)
            rows = n_users if name.startswith('user') else len(keys) if name.startswith('loo') else n_items
            if values.shape != (rows, 3) or not np.isfinite(values).all():
                raise ValueError(f'Invalid feature matrix: {name}')
            self.register_buffer(name+'_input', torch.from_numpy(values), persistent=False)

    def _id_embeddings(self):
        current = torch.cat([self.E_u.weight, self.E_i.weight])
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total/(self.n_layers+1)
        return total[:self.n_users], total[self.n_users:]

    def embeddings(self, need_value=True):
        user, item = self._id_embeddings()
        user_parts, item_parts = [user], [item]
        for axis, alpha in (('n', self.alpha_n), ('v', self.alpha_v)):
            user_parts.append(alpha**.5*self.encoders['user_'+axis](getattr(self, 'user_'+axis+'_input')))
            item_parts.append(alpha**.5*self.encoders['item_'+axis](getattr(self, 'item_'+axis+'_input')))
        return (torch.cat(user_parts, 1), torch.cat(item_parts, 1),
                user.new_zeros((self.n_users, 1)), item.new_zeros((self.n_items, 1)))

    def _pair_scores(self, users, positives, negatives):
        uid, iid = self._id_embeddings()
        keys = users*self.n_items+positives
        positions = torch.searchsorted(self.pair_keys, keys).clamp(max=len(self.pair_keys)-1)
        if not torch.equal(self.pair_keys[positions], keys):
            raise ValueError('Positive absent from training pairs')
        pos = (uid[users]*iid[positives]).sum(1)
        neg = (uid[users]*iid[negatives]).sum(1)
        for axis, alpha in (('n', self.alpha_n), ('v', self.alpha_v)):
            u = self.encoders['user_'+axis](getattr(self, 'user_'+axis+'_input')[users])
            p = self.encoders['item_'+axis](getattr(self, 'loo_'+axis+'_input')[positions])
            n = self.encoders['item_'+axis](getattr(self, 'item_'+axis+'_input')[negatives])
            pos, neg = pos+alpha*(u*p).sum(1), neg+alpha*(u*n).sum(1)
        return pos, neg

    def batch_l2(self, users, positives, negatives):
        tables = [self.E_u(users), self.E_i(positives), self.E_i(negatives)]
        # Shared matrices counted ONCE, never once per user/item occurrence.
        return self.pref_reg*(sum(t.square().sum() for t in tables) +
                              sum(p.square().sum() for p in self.encoders.parameters()))/len(users)

    def _bpr(self, users, positives, negatives, weights=None):
        pos, neg = self._pair_scores(users, positives, negatives)
        row_loss = F.softplus(neg-pos)
        if weights is not None:
            if weights.shape != row_loss.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Positive finite row weights required')
            row_loss = weights*row_loss
        bpr = row_loss.mean()
        return bpr+self.batch_l2(users, positives, negatives), dict(
            bpr=float(bpr.detach()), p_correct=float((pos > neg).float().mean().detach()))

    def bpr_loss(self, users, positives, negatives, weights=None):
        if weights is not None:
            raise ValueError('M2 standalone must use unweighted BPR')
        return self._bpr(users, positives, negatives)

    def weighted_bpr_loss(self, users, positives, negatives, weights):
        if weights is None:
            raise ValueError('M5 requires fixed M4 row weights')
        return self._bpr(users, positives, negatives, weights)

    @torch.no_grad()
    def representation_diagnostics(self):
        result = dict(alpha_n=self.alpha_n, alpha_v=self.alpha_v,
                      shared_nv_parameters=sum(p.numel() for p in self.encoders.parameters()))
        for name, encoder in self.encoders.items():
            values = encoder(getattr(self, name+'_input'))
            result[name+'_mean_norm'] = float(values.norm(dim=1).mean())
            result[name+'_coordinate_std'] = float(values.std(dim=0, unbiased=False).mean())
        return result

    def training_gradient_diagnostics(self):
        return {name+'_last_batch_gradient_norm': float(layer.weight.grad.norm())
                if layer.weight.grad is not None else None for name, layer in self.encoders.items()}
