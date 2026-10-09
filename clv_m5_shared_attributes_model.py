"""Joint attribute/history representation and differentiable weighted LightGCN.

N/V remain observed CLV components, not item-repeat rates or unit prices.
All learned blocks receive the same recommendation gradient. Direct history
excludes the positive item during BPR; graph propagation retains train edges
as in standard LightGCN. No unseen-item cold-start claim is implied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class _EdgeSparseMM(torch.autograd.Function):
    """Exact first-order COO matmul gradients without a dense N-by-N dA.

    For an existing edge (r,c), dL/da_rc = dot(dL/dY[r], X[c]).
    Computing just those dots in chunks bounds workspace by chunk_size*dim.
    Degree-normalization and graph-network gradients remain ordinary autograd
    upstream of the edge-value input. Second derivatives are not supported.
    """
    @staticmethod
    def forward(ctx, indices, values, dense, chunk_size):
        ctx.save_for_backward(indices, values, dense)
        ctx.chunk_size = chunk_size
        n = dense.shape[0]
        adjacency = torch.sparse_coo_tensor(indices, values, (n, n),
                                            check_invariants=False).coalesce()
        return torch.sparse.mm(adjacency, dense)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        indices, values, dense = ctx.saved_tensors
        grad_values = grad_dense = None
        if ctx.needs_input_grad[1]:
            grad_values = torch.empty_like(values)
            for start in range(0, len(values), ctx.chunk_size):
                end = min(start + ctx.chunk_size, len(values))
                rows, cols = indices[:, start:end]
                grad_values[start:end] = (grad_output[rows] * dense[cols]).sum(dim=1)
        if ctx.needs_input_grad[2]:
            n = dense.shape[0]
            transpose = torch.sparse_coo_tensor(indices.flip(0), values, (n, n),
                                                check_invariants=False).coalesce()
            grad_dense = torch.sparse.mm(transpose, grad_output.contiguous())
        return None, grad_values, grad_dense, None


def edge_sparse_mm(indices, values, dense, chunk_size=65536):
    if chunk_size <= 0:
        raise ValueError("엣지 역전파 chunk_size는 양수여야 합니다")
    return _EdgeSparseMM.apply(indices, values, dense, chunk_size)


def build_signals(train, products, n_users, n_items):
    """Only train-derived prices/relations, plus static product metadata."""
    raw_ids = train.groupby("i_idx").i_raw.first().reindex(range(n_items))
    if raw_ids.isna().any() or products.PRODUCT_ID.duplicated().any():
        raise ValueError("상품 mapping 누락 또는 metadata 중복")
    meta = products.set_index("PRODUCT_ID").reindex(raw_ids.to_numpy())
    names = (meta.COMMODITY_DESC.fillna("UNKNOWN").astype(str) + "|"
             + meta.SUB_COMMODITY_DESC.fillna("UNKNOWN").astype(str))
    types, labels = pd.factorize(names, sort=True)
    prices = train.groupby("i_idx").up.median().reindex(range(n_items))
    valid = prices.notna() & np.isfinite(prices) & prices.gt(0)
    price_frame = pd.DataFrame({"price": prices.where(valid), "type": types})
    global_q = (price_frame.price.rank(method="average") - .5) / valid.sum()
    grouped = price_frame.groupby("type").price
    local_q = (grouped.rank(method="average") - .5) / grouped.transform("count")
    price_features = np.column_stack([global_q.fillna(0), local_q.fillna(0), valid]).astype("float32")
    pairs = train.assign(positive_amount=train.v.clip(lower=0)).groupby(["u_idx", "i_idx"], sort=True).agg(
        baskets=("b_raw", "nunique"), amount=("positive_amount", "sum"),
        last=("t", "max"))
    users = pairs.index.get_level_values(0).to_numpy(dtype="int64")
    items = pairs.index.get_level_values(1).to_numpy(dtype="int64")
    user_amount = np.bincount(users, weights=pairs.amount, minlength=n_users)
    shares = np.divide(pairs.amount.to_numpy(), user_amount[users],
                       out=np.zeros(len(users)), where=user_amount[users] > 0)
    end = train.t.max()
    if pd.api.types.is_datetime64_any_dtype(train.t):
        age = (end - pairs["last"]).dt.total_seconds().to_numpy() / 86400
    else:
        age = (end - pairs["last"]).to_numpy(float)
    recent = 1 - np.clip(age / 365, 0, 1)
    log_count = np.log1p(pairs.baskets.to_numpy(float))
    relations = np.column_stack([log_count / max(log_count.max(), 1), shares, recent]).astype("float32")
    return {"edge_users": users, "edge_items": items, "relations": relations,
            "item_types": types.astype("int64"), "price_features": price_features,
            "n_types": len(labels), "audit": {
                "edges": len(users), "item_types": len(labels),
                "metadata_missing_items": int(meta.COMMODITY_DESC.isna().sum()),
                "price_valid_items": int(valid.sum()), "price_source": "train median unit price",
                "history": "uniform unique-item mean; recency only conditions features",
                "graph_inputs": "distinct baskets per pair, positive spend share, 365day recency",
                "all_catalogue_items_preserved": len(types) == n_items}}


class JointAttributeLightGCN(nn.Module):
    def __init__(self, *, n_users, n_items, signals, q_n, q_v, valid,
                 id_dim=64, n_layers=2, pref_reg=.001, eta=.1,
                 epsilon=.1, kappa=.2):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.id_dim, self.n_layers, self.pref_reg = id_dim, n_layers, pref_reg
        self.eta, self.epsilon, self.kappa = eta, epsilon, kappa
        self.E_u, self.E_i = nn.Embedding(n_users, id_dim), nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=.1)
        nn.init.normal_(self.E_i.weight, std=.1)
        self.type_encoder = nn.Embedding(signals["n_types"], 8)
        self.price_encoder = nn.Linear(3, 8)
        self.feature_net = nn.Sequential(nn.Linear(19, 8), nn.Tanh(), nn.Linear(8, 16))
        self.graph_net = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 1))
        for net in (self.feature_net, self.graph_net):
            nn.init.normal_(net[-1].weight, std=.01)
            nn.init.zeros_(net[-1].bias)
        for key in ("edge_users", "edge_items", "item_types"):
            self.register_buffer(key, torch.as_tensor(signals[key], dtype=torch.long))
        for key in ("relations", "price_features"):
            self.register_buffer(key, torch.as_tensor(signals[key], dtype=torch.float32))
        self.register_buffer("valid", torch.as_tensor(valid, dtype=torch.float32))
        context = np.column_stack([np.where(valid, q_n, 0), np.where(valid, q_v, 0)])
        self.register_buffer("context", torch.as_tensor(context, dtype=torch.float32))
        keys = self.edge_users * n_items + self.edge_items
        if len(keys) == 0 or not torch.all(keys[1:] > keys[:-1]):
            raise ValueError("고유 엣지는 정렬되어야 합니다")
        self.register_buffer("edge_keys", keys)
        self.register_buffer("degree", torch.bincount(self.edge_users, minlength=n_users).float())
        shifted = self.edge_items + n_users
        self.register_buffer("indices", torch.stack([
            torch.cat([self.edge_users, shifted]), torch.cat([shifted, self.edge_users])]))

    def attributes(self):
        types = F.normalize(self.type_encoder(self.item_types), dim=1)
        prices = F.normalize(self.price_encoder(self.price_features), dim=1)
        prices = prices * self.price_features[:, 2:3]
        return torch.cat([types, prices], dim=1)

    def graph_weights(self):
        inputs = torch.cat([self.relations, self.context[self.edge_users]], dim=1)
        return torch.exp(self.kappa * torch.tanh(self.graph_net(inputs).squeeze(1)))

    def normalized_edge_values(self):
        weights = self.graph_weights()
        destinations = self.edge_items + self.n_users
        degree = weights.new_zeros(self.n_users + self.n_items)
        degree = degree.index_add(0, self.edge_users, weights).index_add(0, destinations, weights)
        norm = weights / (degree[self.edge_users] * degree[destinations]).clamp_min(1e-12).sqrt()
        return torch.cat([norm, norm])

    def weighted_adjacency(self):
        # Diagnostic/reference representation; propagation uses the edge-value
        # operator below to avoid native sparse-value backward's dense dA.
        n = self.n_users + self.n_items
        return torch.sparse_coo_tensor(self.indices, self.normalized_edge_values(),
                                       (n, n), check_invariants=False).coalesce()

    def id_vectors(self):
        values = self.normalized_edge_values()
        current = torch.cat([self.E_u.weight, self.E_i.weight])
        layers = [current]
        for _ in range(self.n_layers):
            current = edge_sparse_mm(self.indices, values, current)
            layers.append(current)
        return torch.stack(layers).mean(0).split([self.n_users, self.n_items])

    def history_vectors(self, attributes):
        context = self.context[self.edge_users]
        inputs = torch.cat([attributes[self.edge_items], context, self.relations[:, 2:3]], dim=1)
        modulation = self.epsilon * torch.tanh(self.feature_net(inputs))
        modulation = modulation * self.valid[self.edge_users, None]
        messages = attributes[self.edge_items] * (1 + modulation)
        sums = messages.new_zeros(self.n_users, 16).index_add(0, self.edge_users, messages)
        return sums / self.degree[:, None].clamp_min(1), messages

    def leave_one_out(self, profiles, messages, users, positives):
        keys = users * self.n_items + positives
        index = torch.searchsorted(self.edge_keys, keys)
        if torch.any(index >= len(self.edge_keys)) or not torch.equal(self.edge_keys[index], keys):
            raise ValueError("양성상품이 학습이력에 없습니다")
        count = self.degree[users, None]
        output = (profiles[users] * count - messages[index]) / (count - 1).clamp_min(1)
        return torch.where(count > 1, output, torch.zeros_like(output))

    def embeddings(self):
        user_id, item_id = self.id_vectors()
        attributes = self.attributes()
        profiles, _ = self.history_vectors(attributes)
        users = torch.cat([user_id, self.eta * profiles], 1)
        items = torch.cat([item_id, self.eta * attributes], 1)
        # Existing evaluator expects four matrices, even at lambda=0.
        return users, items, users.new_zeros(self.n_users, 1), items.new_zeros(self.n_items, 1)

    def weighted_bpr_loss(self, users, positives, negatives, weights):
        user_id, item_id = self.id_vectors()
        attributes = self.attributes()
        profiles, messages = self.history_vectors(attributes)
        profile = self.leave_one_out(profiles, messages, users, positives)
        user_z = torch.cat([user_id[users], self.eta * profile], 1)
        pos_z = torch.cat([item_id[positives], self.eta * attributes[positives]], 1)
        neg_z = torch.cat([item_id[negatives], self.eta * attributes[negatives]], 1)
        positive = (user_z * pos_z).sum(1)
        negative = (user_z * neg_z).sum(1)
        bpr = (weights * F.softplus(negative - positive)).mean()
        regularization = self.pref_reg * (
            self.E_u(users).square().sum() + self.E_i(positives).square().sum()
            + self.E_i(negatives).square().sum()) / len(users)
        return bpr + regularization, {"bpr": float(bpr.detach()),
                                      "p_correct": float((positive > negative).float().mean())}

    @torch.no_grad()
    def training_gradient_diagnostics(self):
        blocks = {"id_user": self.E_u, "id_item": self.E_i,
                  "shared_type": self.type_encoder, "shared_price": self.price_encoder,
                  "nv_feature": self.feature_net, "nv_graph": self.graph_net}
        return {name + "_gradient_norm": float(sum(
            p.grad.square().sum() for p in module.parameters() if p.grad is not None).sqrt())
            if any(p.grad is not None for p in module.parameters()) else 0.
            for name, module in blocks.items()}

    @torch.no_grad()
    def representation_diagnostics(self):
        weights = self.graph_weights()
        attributes = self.attributes()
        profiles, _ = self.history_vectors(attributes)
        sample = torch.arange(min(1024, len(self.edge_users)), device=weights.device)
        users = self.edge_users[sample]
        context = self.context[users]
        attr_inputs = torch.cat([attributes[self.edge_items[sample]], context,
                                 self.relations[sample, 2:3]], 1)
        graph_inputs = torch.cat([self.relations[sample], context], 1)
        changed_attr, changed_graph = attr_inputs.clone(), graph_inputs.clone()
        changed_attr[:, 16:18] = 1 - context
        changed_graph[:, 3:5] = 1 - context
        return {"graph_weight_min": float(weights.min()), "graph_weight_max": float(weights.max()),
                "graph_weight_cv": float(weights.std(unbiased=False) / weights.mean()),
                "attribute_norm_mean": float(attributes.norm(dim=1).mean()),
                "history_norm_mean": float(profiles.norm(dim=1).mean()),
                "nv_feature_local_sensitivity": float((self.feature_net(changed_attr)
                    - self.feature_net(attr_inputs)).abs().mean()),
                "nv_graph_local_sensitivity": float((self.graph_net(changed_graph)
                    - self.graph_net(graph_inputs)).abs().mean()),
                "sensitivity_is_local_function_probe_not_clv_attribution": True,
                **self.training_gradient_diagnostics()}
