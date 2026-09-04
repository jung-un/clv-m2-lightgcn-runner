"""Sparse single-head GAT with the established CLV level/composition/price input.

Only the propagation rule differs from the matched LightGCN/NGCF screens.  The
binary user-item graph, uniform negatives, plain BPR objective and three CLV
layer-0 coordinates remain fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ngcf_clv_level_composition_price_model import (
    NGCFCLVLevelCompositionPrice,
)


class _WeightedNeighborSum(torch.autograd.Function):
    """Edge-linear aggregation without a differentiable sparse matrix."""

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        receiver: torch.Tensor,
        source: torch.Tensor,
        weights: torch.Tensor,
        n_nodes: int,
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(features, receiver, source, weights)
        ctx.n_nodes = int(n_nodes)
        ctx.chunk_size = int(chunk_size)
        output = features.new_zeros((ctx.n_nodes, features.shape[1]))
        for start in range(0, len(weights), ctx.chunk_size):
            end = min(start + ctx.chunk_size, len(weights))
            messages = features[source[start:end]] * weights[start:end, None]
            output.index_add_(0, receiver[start:end], messages)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        features, receiver, source, weights = ctx.saved_tensors
        grad_features = torch.zeros_like(features) if ctx.needs_input_grad[0] else None
        grad_weights = torch.zeros_like(weights) if ctx.needs_input_grad[3] else None
        for start in range(0, len(weights), ctx.chunk_size):
            end = min(start + ctx.chunk_size, len(weights))
            chunk_receiver = receiver[start:end]
            chunk_source = source[start:end]
            receiver_gradient = grad_output[chunk_receiver]
            if grad_features is not None:
                grad_features.index_add_(
                    0,
                    chunk_source,
                    receiver_gradient * weights[start:end, None],
                )
            if grad_weights is not None:
                grad_weights[start:end] = (
                    receiver_gradient * features[chunk_source]
                ).sum(dim=1)
        return grad_features, None, None, grad_weights, None, None


def _weighted_neighbor_sum(
    features: torch.Tensor,
    receiver: torch.Tensor,
    source: torch.Tensor,
    weights: torch.Tensor,
    n_nodes: int,
    chunk_size: int = 262_144,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features는 [node, dimension]이어야 합니다")
    if receiver.shape != source.shape or receiver.shape != weights.shape:
        raise ValueError("receiver/source/weights shape이 같아야 합니다")
    if chunk_size <= 0:
        raise ValueError("chunk_size는 양수여야 합니다")
    return _WeightedNeighborSum.apply(
        features, receiver, source, weights, int(n_nodes), int(chunk_size)
    )


class SparseSingleHeadGATLayer(nn.Module):
    """One memory-conscious GAT layer over a pre-existing sparse graph."""

    def __init__(self, dimension: int, negative_slope: float = 0.2):
        super().__init__()
        self.dimension = int(dimension)
        self.negative_slope = float(negative_slope)
        self.projection = nn.Linear(dimension, dimension, bias=False)
        self.receiver_attention = nn.Parameter(torch.empty(dimension))
        self.source_attention = nn.Parameter(torch.empty(dimension))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.receiver_attention[None, :])
        nn.init.xavier_uniform_(self.source_attention[None, :])

    def attention_weights(
        self,
        features: torch.Tensor,
        receiver: torch.Tensor,
        source: torch.Tensor,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed = self.projection(features)
        receiver_score = transformed @ self.receiver_attention
        source_score = transformed @ self.source_attention
        logits = F.leaky_relu(
            receiver_score[receiver] + source_score[source],
            negative_slope=self.negative_slope,
        )

        row_max = logits.new_full((n_nodes,), -torch.inf)
        row_max.scatter_reduce_(
            0, receiver, logits, reduce="amax", include_self=True
        )
        stable = torch.exp(logits - row_max[receiver])
        row_sum = logits.new_zeros(n_nodes)
        row_sum.scatter_add_(0, receiver, stable)
        weights = stable / row_sum[receiver].clamp_min(1e-12)
        return receiver, source, weights, transformed

    def forward(
        self,
        features: torch.Tensor,
        receiver: torch.Tensor,
        source: torch.Tensor,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        receiver, source, weights, transformed = self.attention_weights(
            features, receiver, source, n_nodes
        )
        aggregated = _weighted_neighbor_sum(
            transformed, receiver, source, weights, n_nodes
        )
        return F.elu(aggregated), weights


class GATCLVLevelCompositionPrice(NGCFCLVLevelCompositionPrice):
    """GAT@64, GAT@67 or GAT with the same 64+3 CLV layer-0 input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The parent is used only to keep the already-tested CLV coordinates
        # exactly identical.  NGCF propagation parameters are removed before
        # the optimizer can see the model.
        del self.sum_layers
        del self.bi_layers

        graph = self.adj.coalesce()
        if graph.shape != (
            self.n_users + self.n_items,
            self.n_users + self.n_items,
        ):
            raise ValueError("인접행렬 shape이 전체 노드 수와 다릅니다")
        indices = graph.indices()
        self.register_buffer("edge_receiver", indices[0].clone(), persistent=False)
        self.register_buffer("edge_source", indices[1].clone(), persistent=False)
        del self.adj
        self.gat_layers = nn.ModuleList(
            SparseSingleHeadGATLayer(self.input_dim, self.negative_slope)
            for _ in range(self.n_layers)
        )

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def attention_weights(
        self, layer_index: int, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        receiver, source, weights, _ = self.gat_layers[
            layer_index
        ].attention_weights(
            features,
            self.edge_receiver,
            self.edge_source,
            self.n_users + self.n_items,
        )
        return receiver, source, weights

    def propagation_layers(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        user0, item0 = self.layer0_embeddings()
        current = torch.cat([user0, item0], dim=0)
        layers = [current]
        attention_weights = []
        for layer in self.gat_layers:
            current, weights = layer(
                current,
                self.edge_receiver,
                self.edge_source,
                self.n_users + self.n_items,
            )
            layers.append(current)
            attention_weights.append(weights)
        return layers, attention_weights

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        layers, _ = self.propagation_layers()
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor | None) -> float:
            if parameter is None or parameter.grad is None:
                return 0.0
            return float(parameter.grad.norm())

        first = self.gat_layers[0]
        attention_gradient = (
            norm(first.receiver_attention) ** 2
            + norm(first.source_attention) ** 2
        ) ** 0.5
        result = {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
            "gat_projection_layer0_gradient_norm": norm(first.projection.weight),
            "gat_attention_layer0_gradient_norm": attention_gradient,
            "gat_auxiliary_input_column_gradient_norm": (
                float(first.projection.weight.grad[:, self.id_dim :].norm())
                if self.variant == "clv"
                and first.projection.weight.grad is not None
                else 0.0
            ),
        }
        if self.variant == "clv":
            result.update(
                item_relation_projection_gradient_norm=norm(
                    self.item_collaborative_projection.weight
                ),
                item_price_mixer_gradient_norm=norm(self.item_price_logits),
            )
        return result

    @torch.no_grad()
    def _attention_diagnostics(self) -> dict[str, float]:
        layers, attentions = self.propagation_layers()
        del layers
        n_nodes = self.n_users + self.n_items
        degree = torch.bincount(self.edge_receiver, minlength=n_nodes)
        result: dict[str, float] = {}
        for index, weights in enumerate(attentions):
            row_sum = weights.new_zeros(n_nodes)
            row_sum.scatter_add_(0, self.edge_receiver, weights)
            active = degree > 0
            entropy_terms = -weights * torch.log(weights.clamp_min(1e-12))
            entropy = weights.new_zeros(n_nodes)
            entropy.scatter_add_(0, self.edge_receiver, entropy_terms)
            multi = degree > 1
            normalized_entropy = entropy[multi] / torch.log(degree[multi].float())
            row_max = weights.new_zeros(n_nodes)
            row_max.scatter_reduce_(
                0,
                self.edge_receiver,
                weights,
                reduce="amax",
                include_self=True,
            )
            result.update(
                {
                    f"gat_layer{index + 1}_attention_row_sum_max_error": float(
                        (row_sum[active] - 1.0).abs().max()
                    ),
                    f"gat_layer{index + 1}_normalized_attention_entropy_mean": (
                        float(normalized_entropy.mean()) if multi.any() else 0.0
                    ),
                    f"gat_layer{index + 1}_max_attention_mean": float(
                        row_max[active].mean()
                    ),
                }
            )
        return result

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        result: dict[str, float | int | bool | str] = {
            "backbone": "GAT",
            "variant": self.variant,
            "id_dim": self.id_dim,
            "aux_dim": self.aux_dim,
            "layer_input_dim": self.input_dim,
            "final_embedding_dim": self.output_dim,
            "n_layers": self.n_layers,
            "attention_heads": 1,
            "edge_linear_attention_aggregation": True,
            "layer_aggregation": "mean(layer0,layer1,layer2)",
            "binary_graph": True,
            "one_dot_score": True,
            "external_reranking": False,
            "attention_dropout": 0.0,
            "feature_dropout": 0.0,
        }
        result.update(self._attention_diagnostics())
        if self.variant == "clv":
            user_aux = self.user_auxiliary_coordinates()
            item_aux = self.item_auxiliary_coordinates()
            price_weights = torch.softmax(self.item_price_logits, dim=0)
            result.update(
                historical_clv_input=True,
                clv_level_and_nv_composition=True,
                explicit_item_price=True,
                rho_layer0_input_budget=self.rho,
                item_price_budget=self.item_price_budget,
                clv_valid_share=float(self.user_clv_valid.mean()),
                item_economic_valid_share=float(self.item_economic_valid.mean()),
                layer0_user_auxiliary_mean_norm=float(
                    (self.rho**0.5 * user_aux).norm(dim=1).mean()
                ),
                layer0_item_auxiliary_mean_norm=float(
                    (self.rho**0.5 * item_aux).norm(dim=1).mean()
                ),
                first_layer_auxiliary_column_norm=float(
                    self.gat_layers[0]
                    .projection.weight[:, self.id_dim :]
                    .norm()
                ),
                item_relation_projection_norm=float(
                    self.item_collaborative_projection.weight.norm()
                ),
                item_price_weight_overall=float(price_weights[0]),
                item_price_weight_within_category=float(price_weights[1]),
            )
        else:
            result.update(historical_clv_input=False)
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
            "gat_layer1_normalized_attention_entropy_mean",
            "gat_layer2_normalized_attention_entropy_mean",
        )
        return {key: diagnostics[key] for key in keys if key in diagnostics}
