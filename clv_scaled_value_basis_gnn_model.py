"""Matched NGCF, GAT and GraphSAGE models for the M1/M2/M4/M5 screen.

The propagation width is fixed at 67 in every arm.  M1 and M4 use a freely
learned 67-dimensional ID embedding.  M2 and M5 replace three of those
coordinates with the fixed q_C-scaled q_V/item-price RBF basis, leaving 64
trainable ID coordinates.  This makes the representation comparison about the
information occupying the final three coordinates rather than a wider model.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_m5_n_conditioned_value_basis_model import fixed_value_basis
from gat_clv_level_composition_price_model import SparseSingleHeadGATLayer
from graphsage_clv_level_composition_price_model import GraphSAGEMeanLayer


BACKBONES = ("ngcf", "gat", "graphsage")


class CLVScaledValueBasisGNN(nn.Module):
    """One matched GNN backbone with the M2 representation switched on or off."""

    def __init__(
        self,
        *,
        backbone: str,
        m2_active: bool,
        n_users: int,
        n_items: int,
        user_q_v: np.ndarray,
        user_q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        item_price_percentile: np.ndarray,
        item_price_valid: np.ndarray,
        adj: torch.Tensor,
        base_id_dim: int = 64,
        economic_dim: int = 3,
        rho: float = 0.05,
        basis_bandwidth: float = 0.25,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone은 {BACKBONES} 중 하나여야 합니다")
        if min(n_users, n_items, base_id_dim, economic_dim) <= 0:
            raise ValueError("사용자·상품·표현 차원은 양수여야 합니다")
        if economic_dim != 3:
            raise ValueError("이 실험의 고정 RBF 가치기저는 3차원입니다")
        if n_layers != 2:
            raise ValueError("세 GNN 비교는 2층으로 고정합니다")
        if not 0.0 <= rho <= 1.0 or pref_reg < 0.0:
            raise ValueError("rho 또는 pref_reg 설정이 잘못됐습니다")
        if adj.layout != torch.sparse_coo:
            raise ValueError("adj는 sparse COO tensor여야 합니다")
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape이 사용자·상품 수와 다릅니다")

        q_v = np.asarray(user_q_v, dtype=np.float32)
        q_c = np.asarray(user_q_c, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_price = np.asarray(item_price_percentile, dtype=np.float32)
        item_valid = np.asarray(item_price_valid, dtype=bool)
        expected = (
            ("user_q_v", q_v, (n_users,)),
            ("user_q_c", q_c, (n_users,)),
            ("user_clv_valid", user_valid, (n_users,)),
            ("item_price_percentile", item_price, (n_items,)),
            ("item_price_valid", item_valid, (n_items,)),
        )
        for name, values, shape in expected:
            if values.shape != shape:
                raise ValueError(f"{name} shape이 잘못됐습니다")
        if not np.isfinite(q_v).all() or not np.isfinite(q_c).all():
            raise ValueError("q_V·q_C는 모두 유한해야 합니다")
        if np.any((q_v < 0.0) | (q_v > 1.0)) or np.any(
            (q_c < 0.0) | (q_c > 1.0)
        ):
            raise ValueError("q_V·q_C는 [0,1] 범위여야 합니다")
        if np.any(q_v[~user_valid] != 0.0) or np.any(q_c[~user_valid] != 0.0):
            raise ValueError("무효 사용자의 q_V·q_C는 0이어야 합니다")

        user_basis = fixed_value_basis(
            q_v, user_valid, bandwidth=basis_bandwidth
        )
        item_basis = fixed_value_basis(
            item_price, item_valid, bandwidth=basis_bandwidth
        )

        self.backbone = backbone
        self.m2_active = bool(m2_active)
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.base_id_dim = int(base_id_dim)
        self.economic_dim = int(economic_dim)
        self.total_dim = self.base_id_dim + self.economic_dim
        self.trainable_id_dim = (
            self.base_id_dim if self.m2_active else self.total_dim
        )
        self.rho = float(rho if self.m2_active else 0.0)
        self.basis_bandwidth = float(basis_bandwidth)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.negative_slope = float(negative_slope)

        self.E_u = nn.Embedding(n_users, self.trainable_id_dim)
        self.E_i = nn.Embedding(n_items, self.trainable_id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.register_buffer(
            "user_value_coordinates",
            torch.from_numpy(q_c[:, None] * user_basis),
            persistent=False,
        )
        self.register_buffer(
            "item_value_coordinates", torch.from_numpy(item_basis), persistent=False
        )
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )

        graph = adj.coalesce()
        if backbone == "ngcf":
            self.register_buffer("adj", graph, persistent=False)
            self.sum_layers = nn.ModuleList(
                nn.Linear(self.total_dim, self.total_dim)
                for _ in range(self.n_layers)
            )
            self.bi_layers = nn.ModuleList(
                nn.Linear(self.total_dim, self.total_dim)
                for _ in range(self.n_layers)
            )
            for layer in [*self.sum_layers, *self.bi_layers]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif backbone == "gat":
            indices = graph.indices()
            self.register_buffer("edge_receiver", indices[0].clone(), persistent=False)
            self.register_buffer("edge_source", indices[1].clone(), persistent=False)
            self.gat_layers = nn.ModuleList(
                SparseSingleHeadGATLayer(self.total_dim, self.negative_slope)
                for _ in range(self.n_layers)
            )
        else:
            n_nodes = self.n_users + self.n_items
            indices = graph.indices()
            receiver = indices[0]
            degree = torch.bincount(receiver, minlength=n_nodes)
            if (degree == 0).any():
                raise ValueError("GraphSAGE 평균전파에 고립 노드가 있습니다")
            values = degree[receiver].to(dtype=torch.float32).reciprocal()
            with torch.sparse.check_sparse_tensor_invariants(False):
                mean_adjacency = torch.sparse_coo_tensor(
                    indices.clone(), values, graph.shape, device=indices.device
                ).coalesce()
            self.register_buffer("mean_adjacency", mean_adjacency, persistent=False)
            self.sage_layers = nn.ModuleList(
                GraphSAGEMeanLayer(self.total_dim) for _ in range(self.n_layers)
            )

    @property
    def output_dim(self) -> int:
        if self.backbone == "ngcf":
            return self.total_dim * (self.n_layers + 1)
        return self.total_dim

    def layer0_embeddings(
        self, *, m2_active: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = self.m2_active if m2_active is None else bool(m2_active)
        if active and not self.m2_active:
            raise ValueError("M1·M4 모형에서 M2 입력을 사후에 켤 수 없습니다")
        if not self.m2_active:
            return self.E_u.weight, self.E_i.weight

        scale = math.sqrt(self.rho)
        if active:
            user_aux = scale * self.user_value_coordinates
            item_aux = scale * self.item_value_coordinates
        else:
            user_aux = self.E_u.weight.new_zeros((self.n_users, self.economic_dim))
            item_aux = self.E_i.weight.new_zeros((self.n_items, self.economic_dim))
        return (
            torch.cat([self.E_u.weight, user_aux], dim=1),
            torch.cat([self.E_i.weight, item_aux], dim=1),
        )

    def propagation_layers(
        self, *, m2_active: bool | None = None
    ) -> list[torch.Tensor]:
        user0, item0 = self.layer0_embeddings(m2_active=m2_active)
        current = torch.cat([user0, item0], dim=0)
        layers = [current]
        if self.backbone == "ngcf":
            for sum_layer, bi_layer in zip(
                self.sum_layers, self.bi_layers, strict=True
            ):
                neighbour = torch.sparse.mm(self.adj, current)
                current = F.leaky_relu(
                    sum_layer(current + neighbour)
                    + bi_layer(current * neighbour),
                    negative_slope=self.negative_slope,
                )
                current = F.normalize(current, p=2, dim=1, eps=1e-12)
                layers.append(current)
        elif self.backbone == "gat":
            for layer in self.gat_layers:
                current, _ = layer(
                    current,
                    self.edge_receiver,
                    self.edge_source,
                    self.n_users + self.n_items,
                )
                layers.append(current)
        else:
            for layer in self.sage_layers:
                current = layer(current, self.mean_adjacency)
                layers.append(current)
        return layers

    def propagated_embeddings(
        self, *, m2_active: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layers = self.propagation_layers(m2_active=m2_active)
        if self.backbone == "ngcf":
            final = torch.cat(layers, dim=1)
        else:
            final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]

    def embeddings(self, need_value: bool = True):
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def sampled_l2(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if self.pref_reg <= 0.0:
            return self.E_u.weight.new_zeros(())
        if negatives.ndim != 2:
            raise ValueError("negatives는 [batch,K] shape이어야 합니다")
        negative_mean = (
            self.E_i.weight[negatives].pow(2).sum(dim=2).mean(dim=1).sum()
        )
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + negative_mean
        ) / max(len(users), 1)

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor | None) -> float:
            if parameter is None or parameter.grad is None:
                return 0.0
            return float(parameter.grad.norm())

        result = {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
        }
        if self.backbone == "ngcf":
            result.update(
                backbone_layer0_gradient_norm=(
                    norm(self.sum_layers[0].weight) ** 2
                    + norm(self.bi_layers[0].weight) ** 2
                )
                ** 0.5,
                auxiliary_input_column_gradient_norm=(
                    float(
                        torch.cat(
                            [
                                self.sum_layers[0].weight.grad[:, -self.economic_dim :],
                                self.bi_layers[0].weight.grad[:, -self.economic_dim :],
                            ],
                            dim=1,
                        ).norm()
                    )
                    if self.m2_active
                    and self.sum_layers[0].weight.grad is not None
                    and self.bi_layers[0].weight.grad is not None
                    else 0.0
                ),
            )
        elif self.backbone == "gat":
            first = self.gat_layers[0]
            result.update(
                backbone_layer0_gradient_norm=norm(first.projection.weight),
                attention_layer0_gradient_norm=(
                    norm(first.receiver_attention) ** 2
                    + norm(first.source_attention) ** 2
                )
                ** 0.5,
                auxiliary_input_column_gradient_norm=(
                    float(first.projection.weight.grad[:, -self.economic_dim :].norm())
                    if self.m2_active and first.projection.weight.grad is not None
                    else 0.0
                ),
            )
        else:
            first = self.sage_layers[0].projection.weight
            result.update(
                backbone_layer0_gradient_norm=norm(first),
                auxiliary_input_column_gradient_norm=(
                    float(
                        torch.cat(
                            [
                                first.grad[:, -self.economic_dim :],
                                first.grad[:, self.total_dim - self.economic_dim : self.total_dim],
                            ],
                            dim=1,
                        ).norm()
                    )
                    if self.m2_active and first.grad is not None
                    else 0.0
                ),
            )
        return result

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        result: dict[str, float | int | bool | str] = {
            "backbone": self.backbone,
            "m2_active": self.m2_active,
            "trainable_id_dim": self.trainable_id_dim,
            "economic_dim": self.economic_dim if self.m2_active else 0,
            "matched_total_input_dim": self.total_dim,
            "final_embedding_dim": self.output_dim,
            "n_layers": self.n_layers,
            "rho": self.rho,
            "binary_graph": True,
            "external_reranking": False,
            "joint_end_to_end_training": True,
            "m3_edge_weight": False,
        }
        if self.m2_active:
            user_aux = math.sqrt(self.rho) * self.user_value_coordinates
            item_aux = math.sqrt(self.rho) * self.item_value_coordinates
            result.update(
                historical_clv_input=True,
                q_c_definition="midrank percentile of raw n_u*v_u",
                q_c_role="scales the complete user value-position basis",
                q_v_role="position in fixed low/mid/high value basis",
                separate_q_n_gate=False,
                item_role="amount-percentile position in the same fixed basis",
                basis_bandwidth=self.basis_bandwidth,
                user_auxiliary_mean_norm=float(user_aux.norm(dim=1).mean()),
                item_auxiliary_mean_norm=float(item_aux.norm(dim=1).mean()),
                clv_valid_share=float(self.user_clv_valid.mean()),
            )
        else:
            result.update(
                historical_clv_input=False,
                baseline_capacity_control="67-dimensional trainable ID embedding",
            )
        return result
