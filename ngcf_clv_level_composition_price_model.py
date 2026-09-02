"""NGCF using the fixed CLV level/composition/price layer-0 coordinates.

The CLV arm changes only the node representation.  It keeps the binary graph,
uniform negatives and one plain BPR objective used by the surrounding screen.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCFCLVLevelCompositionPrice(nn.Module):
    """One NGCF for either an ID-only or a 64+3 CLV representation arm."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        adj: torch.Tensor,
        id_dim: int,
        variant: str,
        q_n: np.ndarray | None = None,
        q_v: np.ndarray | None = None,
        q_c: np.ndarray | None = None,
        user_clv_valid: np.ndarray | None = None,
        item_economic_features: np.ndarray | None = None,
        item_economic_valid: np.ndarray | None = None,
        rho: float = 0.05,
        item_price_budget: float = 0.25,
        n_layers: int = 2,
        negative_slope: float = 0.2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if variant not in {"id", "clv"}:
            raise ValueError("variant는 id 또는 clv여야 합니다")
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 차원은 양수여야 합니다")
        if n_layers != 2:
            raise ValueError("이번 NGCF screen은 2층으로 고정합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if not 0.0 <= item_price_budget <= 1.0:
            raise ValueError("가격 좌표 예산은 0 이상 1 이하여야 합니다")
        if negative_slope != 0.2:
            raise ValueError("LeakyReLU 기울기는 0.2로 고정합니다")
        if pref_reg < 0:
            raise ValueError("pref_reg는 음수일 수 없습니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.variant = variant
        self.rho = float(rho if variant == "clv" else 0.0)
        self.item_price_budget = float(item_price_budget)
        self.n_layers = int(n_layers)
        self.negative_slope = float(negative_slope)
        self.pref_reg = float(pref_reg)
        self.aux_dim = 3 if variant == "clv" else 0
        self.input_dim = self.id_dim + self.aux_dim

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.sum_layers = nn.ModuleList(
            nn.Linear(self.input_dim, self.input_dim) for _ in range(n_layers)
        )
        self.bi_layers = nn.ModuleList(
            nn.Linear(self.input_dim, self.input_dim) for _ in range(n_layers)
        )
        for layer in [*self.sum_layers, *self.bi_layers]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.register_buffer("adj", adj.coalesce(), persistent=False)
        if variant == "clv":
            self._register_clv_inputs(
                q_n=q_n,
                q_v=q_v,
                q_c=q_c,
                user_clv_valid=user_clv_valid,
                item_economic_features=item_economic_features,
                item_economic_valid=item_economic_valid,
            )
            self.item_collaborative_projection = nn.Linear(id_dim, 2, bias=False)
            nn.init.normal_(self.item_collaborative_projection.weight, std=0.02)
            self.item_price_logits = nn.Parameter(torch.zeros(2))

    def _register_clv_inputs(
        self,
        *,
        q_n,
        q_v,
        q_c,
        user_clv_valid,
        item_economic_features,
        item_economic_valid,
    ) -> None:
        if any(value is None for value in (q_n, q_v, q_c, user_clv_valid)):
            raise ValueError("CLV arm에는 q_n/q_v/q_c/valid가 필요합니다")
        if item_economic_features is None or item_economic_valid is None:
            raise ValueError("CLV arm에는 상품 가격 입력과 유효성 표시가 필요합니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        q_c = np.asarray(q_c, dtype=np.float32)
        valid = np.asarray(user_clv_valid, dtype=bool)
        item_features = np.asarray(item_economic_features, dtype=np.float32)
        item_valid = np.asarray(item_economic_valid, dtype=bool)
        if any(x.shape != (self.n_users,) for x in (q_n, q_v, q_c, valid)):
            raise ValueError("사용자 CLV 입력 shape이 n_users와 다릅니다")
        if item_features.shape != (self.n_items, 2):
            raise ValueError("상품 경제 입력은 [n_items, 2]여야 합니다")
        if item_valid.shape != (self.n_items,):
            raise ValueError("상품 경제 입력 유효성 shape이 n_items와 다릅니다")
        if not all(np.isfinite(x).all() for x in (q_n, q_v, q_c, item_features)):
            raise ValueError("CLV·상품 경제 입력은 모두 유한해야 합니다")
        for name, values in (("q_n", q_n), ("q_v", q_v), ("q_c", q_c)):
            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{name}은 [0,1] 범위여야 합니다")
        if any(np.any(values[~valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 CLV 입력은 0이어야 합니다")
        if float(np.max(np.abs(item_features), initial=0.0)) > 1.0 + 1e-6:
            raise ValueError("상품 경제 입력은 [-1,1] 범위여야 합니다")

        self.register_buffer("q_n", torch.from_numpy(q_n.copy()), persistent=False)
        self.register_buffer("q_v", torch.from_numpy(q_v.copy()), persistent=False)
        self.register_buffer("q_c", torch.from_numpy(q_c.copy()), persistent=False)
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_economic_features",
            torch.from_numpy(item_features.copy()),
            persistent=False,
        )
        self.register_buffer(
            "item_economic_valid",
            torch.from_numpy(item_valid.astype(np.float32)),
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        return self.input_dim * (self.n_layers + 1)

    def user_auxiliary_coordinates(self) -> torch.Tensor:
        if self.variant != "clv":
            return self.E_u.weight.new_zeros((self.n_users, 0))
        composition = self.q_n - self.q_v
        relation = self.q_c[:, None] * F.normalize(
            torch.stack([self.q_c, composition], dim=1),
            p=2,
            dim=1,
            eps=1e-12,
        )
        price_preference = self.q_c * (2.0 * self.q_v - 1.0)
        coordinates = torch.cat(
            [
                math.sqrt(1.0 - self.item_price_budget) * relation,
                math.sqrt(self.item_price_budget) * price_preference[:, None],
            ],
            dim=1,
        )
        return self.user_clv_valid[:, None] * coordinates

    def item_auxiliary_coordinates(self) -> torch.Tensor:
        if self.variant != "clv":
            return self.E_i.weight.new_zeros((self.n_items, 0))
        relation = F.normalize(
            self.item_collaborative_projection(self.E_i.weight),
            p=2,
            dim=1,
            eps=1e-12,
        )
        price_weights = torch.softmax(self.item_price_logits, dim=0)
        price = (
            (self.item_economic_features * price_weights[None, :]).sum(
                dim=1, keepdim=True
            )
            * self.item_economic_valid[:, None]
        )
        return torch.cat(
            [
                math.sqrt(1.0 - self.item_price_budget) * relation,
                math.sqrt(self.item_price_budget) * price,
            ],
            dim=1,
        )

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "id":
            return self.E_u.weight, self.E_i.weight
        scale = math.sqrt(self.rho)
        return (
            torch.cat([self.E_u.weight, scale * self.user_auxiliary_coordinates()], 1),
            torch.cat([self.E_i.weight, scale * self.item_auxiliary_coordinates()], 1),
        )

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user0, item0 = self.layer0_embeddings()
        current = torch.cat([user0, item0], dim=0)
        layers = [current]
        for sum_layer, bi_layer in zip(self.sum_layers, self.bi_layers, strict=True):
            neighbour = torch.sparse.mm(self.adj, current)
            current = F.leaky_relu(
                sum_layer(current + neighbour) + bi_layer(current * neighbour),
                negative_slope=self.negative_slope,
            )
            current = F.normalize(current, p=2, dim=1, eps=1e-12)
            layers.append(current)
        final = torch.cat(layers, dim=1)
        return final[: self.n_users], final[self.n_users :]

    def embeddings(self, need_value: bool = True):
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        sampled = (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / max(len(users), 1)
        return self.pref_reg * sampled

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        if float(lam) != 0.0:
            raise ValueError("M2 표현 실험에 외부 점수 보정을 넣을 수 없습니다")
        user, item = self.propagated_embeddings()
        selected = user[users]
        positive_score = (selected * item[positives]).sum(dim=1)
        negative_score = (selected * item[negatives]).sum(dim=1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "objective": "plain_bpr",
                "p_correct": float((positive_score > negative_score).float().mean()),
            }
        return loss, diagnostics

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor | None) -> float:
            if parameter is None or parameter.grad is None:
                return 0.0
            return float(parameter.grad.norm())

        first_sum = self.sum_layers[0].weight
        first_bi = self.bi_layers[0].weight
        result = {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
            "ngcf_sum_layer0_gradient_norm": norm(first_sum),
            "ngcf_bi_layer0_gradient_norm": norm(first_bi),
        }
        if self.variant == "clv":
            result.update(
                item_relation_projection_gradient_norm=norm(
                    self.item_collaborative_projection.weight
                ),
                item_price_mixer_gradient_norm=norm(self.item_price_logits),
                ngcf_auxiliary_input_column_gradient_norm=float(
                    torch.cat(
                        [first_sum.grad[:, self.id_dim :], first_bi.grad[:, self.id_dim :]],
                        dim=1,
                    ).norm()
                )
                if first_sum.grad is not None and first_bi.grad is not None
                else 0.0,
            )
        return result

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        keys = (
            "layer0_user_auxiliary_mean_norm",
            "layer0_item_auxiliary_mean_norm",
            "first_layer_auxiliary_column_norm",
            "item_price_weight_overall",
            "item_price_weight_within_category",
        )
        return {key: diagnostics[key] for key in keys if key in diagnostics}

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        result: dict[str, float | int | bool | str] = {
            "backbone": "NGCF",
            "variant": self.variant,
            "id_dim": self.id_dim,
            "aux_dim": self.aux_dim,
            "layer_input_dim": self.input_dim,
            "final_embedding_dim": self.output_dim,
            "n_layers": self.n_layers,
            "binary_graph": True,
            "one_dot_score": True,
            "external_reranking": False,
            "node_dropout": 0.0,
            "message_dropout": 0.0,
        }
        if self.variant == "clv":
            user_aux = self.user_auxiliary_coordinates()
            item_aux = self.item_auxiliary_coordinates()
            price_weights = torch.softmax(self.item_price_logits, dim=0)
            first_aux = torch.cat(
                [
                    self.sum_layers[0].weight[:, self.id_dim :],
                    self.bi_layers[0].weight[:, self.id_dim :],
                ],
                dim=1,
            )
            result.update(
                historical_clv_input=True,
                clv_level_and_nv_composition=True,
                explicit_item_price=True,
                rho_layer0_input_budget=self.rho,
                item_price_budget=self.item_price_budget,
                clv_valid_share=float(self.user_clv_valid.mean()),
                item_economic_valid_share=float(self.item_economic_valid.mean()),
                layer0_user_auxiliary_mean_norm=float(
                    (math.sqrt(self.rho) * user_aux).norm(dim=1).mean()
                ),
                layer0_item_auxiliary_mean_norm=float(
                    (math.sqrt(self.rho) * item_aux).norm(dim=1).mean()
                ),
                first_layer_auxiliary_column_norm=float(first_aux.norm()),
                item_relation_projection_norm=float(
                    self.item_collaborative_projection.weight.norm()
                ),
                item_price_weight_overall=float(price_weights[0]),
                item_price_weight_within_category=float(price_weights[1]),
            )
        else:
            result.update(historical_clv_input=False)
        return result
