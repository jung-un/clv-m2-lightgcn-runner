"""Train-only shared feature embeddings plus separately penalized ID residuals.

User N/V are historical CLV components, not item-repeat rates or future CLV.
All parameters enter layer 0 and receive the same plain BPR gradient.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from clv_m5_n_conditioned_value_basis_model import fixed_value_basis
from lightgcn_clv_v3 import item_value_features

AXES = ('user_n', 'user_v', 'item_price', 'item_within')


def centered_basis(values, valid, bandwidth=.25):
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError('Axis values/mask shape mismatch')
    basis = fixed_value_basis(np.where(valid, values, .5), valid, bandwidth=bandwidth)
    center = basis[valid].mean(0, dtype=np.float64) if valid.any() else np.zeros(3)
    result = np.where(valid[:, None], basis-center, 0.).astype(np.float32)
    return result, dict(valid_count=int(valid.sum()), unique_valid=int(np.unique(values[valid]).size),
                       center=center.tolist(), centered_rms=float(np.sqrt(np.mean(result**2))))


def build_features(train, *, n_users, n_items, q_n, q_v, n_valid, v_valid, bandwidth=.25):
    if train.empty or not {'u_idx', 'i_idx', 'up', 'cat_idx'}.issubset(train.columns):
        raise ValueError('Nonempty TRAIN u_idx/i_idx/up/cat_idx required')
    for column, size in (('u_idx', n_users), ('i_idx', n_items)):
        ids = train[column].to_numpy()
        if not np.isfinite(ids).all() or np.any((ids < 0) | (ids >= size) | (ids != np.floor(ids))):
            raise ValueError('Training catalog index out of range')
    if any(np.asarray(x).shape != (n_users,) for x in (q_n, q_v, n_valid, v_valid)):
        raise ValueError('User input size mismatch')
    clean = train[['i_idx', 'up', 'cat_idx']].copy()
    clean['up'] = clean.up.where(np.isfinite(clean.up) & (clean.up > 0))
    cat_ok = np.isfinite(clean.cat_idx) & (clean.cat_idx >= 0) & (clean.cat_idx == np.floor(clean.cat_idx))
    clean['cat_idx'] = clean.cat_idx.where(cat_ok, -1)
    prices, categories = item_value_features(clean, n_items, report=False)
    med = clean.groupby('i_idx').up.median()
    price_valid = np.zeros(n_items, bool)
    price_valid[med.index.to_numpy(int)] = np.isfinite(med.to_numpy())
    inputs = ((q_n, n_valid), (q_v, v_valid), (prices[:, 0], price_valid),
              (prices[:, 1], price_valid & (categories >= 0)))
    features, diagnostics = {}, {}
    for name, (values, valid) in zip(AXES, inputs):
        features[name], diagnostics[name] = centered_basis(values, valid, bandwidth)
    diagnostics.update(bandwidth=bandwidth, source='training rows only; fixed during training/evaluation',
        price_definition='positive finite unit-price median; global and within-category percentile',
        missing_policy='zero axis input, ID retained; genuine N=0 and ties retained',
        item_buyer_n_used=False, user_n_and_v_used=True,
        self_inclusion='training-observed features; no claim that all self-inclusion is removed')
    return dict(features, diagnostics=diagnostics)


class SharedFeatureResidualLightGCN(nn.Module):
    supports_clv_score_split = False

    def __init__(self, *, n_users, n_items, adj, features, id_dim=64, n_layers=2,
                 user_l2=.001, item_l2=.001, axis_l2=None):
        super().__init__()
        axis_l2 = dict.fromkeys(AXES, .001) if axis_l2 is None else dict(axis_l2)
        if min(n_users, n_items, id_dim) <= 0 or n_layers < 0:
            raise ValueError('Invalid model dimensions')
        if set(axis_l2) != set(AXES) or not all(np.isfinite(x) and x > 0
                for x in (user_l2, item_l2, *axis_l2.values())):
            raise ValueError('Positive finite residual and four-axis L2 required')
        if adj.layout != torch.sparse_coo or adj.shape != (n_users+n_items,)*2:
            raise ValueError('Binary graph adjacency shape/layout mismatch')
        self.n_users, self.n_items = n_users, n_items
        self.id_dim, self.n_layers = id_dim, n_layers
        self.user_l2, self.item_l2, self.axis_l2 = user_l2, item_l2, axis_l2
        self.disabled_axes = ()  # Evaluation-only removal diagnostic, never a training condition.
        self.E_u, self.E_i = nn.Embedding(n_users, id_dim), nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=.1)
        nn.init.normal_(self.E_i.weight, std=.1)
        # Allocate AFTER ID initialization: exactly the matched M1 ID random draws.
        self.encoders = nn.ModuleDict({name: nn.Linear(3, id_dim, bias=False) for name in AXES})
        for name, layer in self.encoders.items():
            nn.init.zeros_(layer.weight)
            x = np.asarray(features[name], np.float32)
            rows = n_users if name.startswith('user') else n_items
            if x.shape != (rows, 3) or not np.isfinite(x).all():
                raise ValueError(f'Invalid centered input: {name}')
            self.register_buffer(name+'_input', torch.from_numpy(x.copy()), persistent=False)
        self.register_buffer('adj', adj.coalesce(), persistent=False)

    def embeddings(self, need_value=True):
        user, item = self.E_u.weight, self.E_i.weight
        for name, layer in self.encoders.items():
            if name in self.disabled_axes:
                continue
            feature = layer(getattr(self, name+'_input'))
            if name.startswith('user'):
                user = user+feature
            else:
                item = item+feature
        current = total = torch.cat([user, item])
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total+current
        total = total/(self.n_layers+1)
        return (total[:self.n_users], total[self.n_users:],
                total.new_zeros((self.n_users, 1)), total.new_zeros((self.n_items, 1)))

    def _pair_scores(self, users, positives, negatives):
        u, i = self.embeddings()[:2]
        return (u[users]*i[positives]).sum(1), (u[users]*i[negatives]).sum(1)

    def batch_l2(self, users, positives, negatives):
        residual = (self.user_l2*self.E_u(users).square().sum()
                    + self.item_l2*(self.E_i(positives).square().sum()
                                    + self.E_i(negatives).square().sum()))/len(users)
        shared = sum(self.axis_l2[name]*layer.weight.square().sum()
                     for name, layer in self.encoders.items())
        return residual+shared  # Shared matrices are NOT divided by batch size.

    def bpr_loss(self, users, positives, negatives, weights=None):
        if weights is not None or self.disabled_axes:
            raise ValueError('M2 training requires plain BPR and both N/V axes')
        pos, neg = self._pair_scores(users, positives, negatives)
        bpr = F.softplus(neg-pos).mean()
        return bpr+self.batch_l2(users, positives, negatives), dict(
            bpr=float(bpr.detach()), p_correct=float((pos > neg).float().mean().detach()))

    @torch.no_grad()
    def representation_diagnostics(self):
        result = dict(shared_parameters=sum(p.numel() for p in self.encoders.parameters()),
                      user_residual_mean_norm=float(self.E_u.weight.norm(dim=1).mean()),
                      item_residual_mean_norm=float(self.E_i.weight.norm(dim=1).mean()))
        for name, layer in self.encoders.items():
            output = layer(getattr(self, name+'_input'))
            result[name+'_mean_norm'] = float(output.norm(dim=1).mean())
            result[name+'_coordinate_std'] = float(output.std(dim=0, unbiased=False).mean())
            result[name+'_l2_penalty'] = float(self.axis_l2[name]*layer.weight.square().sum())
        return result
