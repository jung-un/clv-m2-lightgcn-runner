"""LightGCN whose only CLV intervention is a trainable weighted graph."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SparseSymmetricMMWithScalar(torch.autograd.Function):
    """Sparse matrix multiplication with an explicit scalar-value derivative.

    PyTorch's generic sparse-value backward materialises a dense adjacency
    gradient for this operation.  The graph depends on only one scalar, so its
    gradient can instead be contracted edge by edge without that dense matrix.
    """

    @staticmethod
    def forward(ctx, indices, values, value_derivative, x, alpha, size):
        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            size,
            device=values.device,
            check_invariants=False,
        ).coalesce()
        derivative_adjacency = torch.sparse_coo_tensor(
            indices,
            value_derivative,
            size,
            device=values.device,
            check_invariants=False,
        ).coalesce()
        ctx.save_for_backward(
            adjacency.indices(),
            adjacency.values(),
            derivative_adjacency.values(),
            x,
        )
        ctx.size = size
        return torch.sparse.mm(adjacency, x)

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, value_derivative, x = ctx.saved_tensors
        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            ctx.size,
            device=values.device,
            is_coalesced=True,
            check_invariants=False,
        )
        grad_x = torch.sparse.mm(adjacency, grad_output)
        rows, columns = indices
        grad_alpha = values.new_zeros(())
        chunk_size = 131_072
        for start in range(0, len(values), chunk_size):
            stop = min(start + chunk_size, len(values))
            edge_dot = (
                grad_output[rows[start:stop]] * x[columns[start:stop]]
            ).sum(dim=1)
            grad_alpha = grad_alpha + (
                value_derivative[start:stop] * edge_dot
            ).sum()
        return None, None, None, grad_x, grad_alpha, None


class CLVLiftGraphLightGCN(nn.Module):
    """Apply CLV-group item lift to observed user-item graph edges."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        edge_users: np.ndarray,
        edge_items: np.ndarray,
        edge_signal: np.ndarray,
        embedding_dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        users = np.asarray(edge_users, dtype=np.int64)
        items = np.asarray(edge_items, dtype=np.int64)
        signal = np.asarray(edge_signal, dtype=np.float32)
        if users.shape != items.shape or users.shape != signal.shape:
            raise ValueError("edge arrays must have the same shape")
        if users.ndim != 1 or len(users) == 0 or not np.isfinite(signal).all():
            raise ValueError("edge arrays must be non-empty, one-dimensional and finite")
        if users.min() < 0 or users.max() >= n_users:
            raise ValueError("edge user index out of range")
        if items.min() < 0 or items.max() >= n_items:
            raise ValueError("edge item index out of range")
        if len(np.unique(users * np.int64(n_items) + items)) != len(users):
            raise ValueError("edge arrays must contain unique user-item pairs")
        if (
            embedding_dim <= 0
            or n_layers < 0
            or pref_reg < 0
            or not 0 < alpha_init < 1
        ):
            raise ValueError("invalid model configuration")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.E_u = nn.Embedding(n_users, embedding_dim)
        self.E_i = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.raw_alpha = nn.Parameter(
            torch.tensor(
                math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32
            )
        )
        self.register_buffer("edge_users", torch.from_numpy(users))
        self.register_buffer("edge_items", torch.from_numpy(items))
        self.register_buffer("edge_signal", torch.from_numpy(signal))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def _edge_values_and_derivative(self):
        alpha = self.alpha().detach()
        raw_weight = torch.exp(alpha * self.edge_signal)
        raw_derivative = raw_weight * self.edge_signal
        user_degree = raw_weight.new_zeros(self.n_users)
        item_degree = raw_weight.new_zeros(self.n_items)
        user_degree_derivative = raw_weight.new_zeros(self.n_users)
        item_degree_derivative = raw_weight.new_zeros(self.n_items)
        user_degree.scatter_add_(0, self.edge_users, raw_weight)
        item_degree.scatter_add_(0, self.edge_items, raw_weight)
        user_degree_derivative.scatter_add_(
            0, self.edge_users, raw_derivative
        )
        item_degree_derivative.scatter_add_(
            0, self.edge_items, raw_derivative
        )
        normalised = raw_weight / torch.sqrt(
            user_degree[self.edge_users] * item_degree[self.edge_items]
        ).clamp_min(1e-12)
        log_derivative = (
            self.edge_signal
            - 0.5
            * user_degree_derivative[self.edge_users]
            / user_degree[self.edge_users].clamp_min(1e-12)
            - 0.5
            * item_degree_derivative[self.edge_items]
            / item_degree[self.edge_items].clamp_min(1e-12)
        )
        normalised_derivative = normalised * log_derivative
        item_nodes = self.edge_items + self.n_users
        indices = torch.stack(
            [
                torch.cat([self.edge_users, item_nodes]),
                torch.cat([item_nodes, self.edge_users]),
            ]
        )
        values = torch.cat([normalised, normalised])
        value_derivative = torch.cat(
            [normalised_derivative, normalised_derivative]
        )
        return indices, values, value_derivative

    def weighted_adjacency(self) -> torch.Tensor:
        indices, values, _ = self._edge_values_and_derivative()
        return torch.sparse_coo_tensor(
            indices,
            values,
            (self.n_users + self.n_items, self.n_users + self.n_items),
            device=values.device,
            check_invariants=False,
        ).coalesce()

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices, values, value_derivative = self._edge_values_and_derivative()
        size = (self.n_users + self.n_items, self.n_users + self.n_items)
        adjacency = None
        if not torch.is_grad_enabled():
            adjacency = torch.sparse_coo_tensor(
                indices,
                values,
                size,
                device=values.device,
                check_invariants=False,
            ).coalesce()
        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            if adjacency is None:
                current = _SparseSymmetricMMWithScalar.apply(
                    indices,
                    values,
                    value_derivative,
                    current,
                    self.alpha(),
                    size,
                )
            else:
                current = torch.sparse.mm(adjacency, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        user, item = self.propagate()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, w=None):
        if w is not None:
            raise ValueError("M4 sample weights are not part of this M3 model")
        if float(lam) != 0.0:
            raise ValueError("external score lambda is not part of this M3 model")
        user, item = self.propagate()
        positive_score = (user[users] * item[positives]).sum(1)
        negative_score = (user[users] * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def graph_diagnostics(self) -> dict[str, float]:
        adjacency = self.weighted_adjacency()
        values = adjacency.values()[: len(self.edge_users)]
        return {
            "learned_alpha": float(self.alpha()),
            "edge_signal_mean": float(self.edge_signal.mean()),
            "edge_signal_std": float(self.edge_signal.std(unbiased=False)),
            "normalised_edge_weight_mean": float(values.mean()),
            "normalised_edge_weight_std": float(values.std(unbiased=False)),
        }
