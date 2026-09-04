"""GraphSAGE with the established CLV level/composition/price input.

Only the propagation backbone changes. The binary graph, uniform negatives,
plain BPR objective and the three CLV layer-0 coordinates stay fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ngcf_clv_level_composition_price_model import (
    NGCFCLVLevelCompositionPrice,
)


class GraphSAGEMeanLayer(nn.Module):
    """Canonical mean aggregation with explicit self and neighbor paths."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)
        self.projection = nn.Linear(2 * self.dimension, self.dimension)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, features: torch.Tensor, mean_adjacency: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise ValueError("features shape이 GraphSAGE 차원과 다릅니다")
        if mean_adjacency.shape != (features.shape[0], features.shape[0]):
            raise ValueError("mean_adjacency shape이 전체 노드 수와 다릅니다")
        neighbor_mean = torch.sparse.mm(mean_adjacency, features)
        updated = F.elu(
            self.projection(torch.cat([features, neighbor_mean], dim=1))
        )
        return F.normalize(updated, p=2, dim=1, eps=1e-12)


class GraphSAGECLVLevelCompositionPrice(NGCFCLVLevelCompositionPrice):
    """GraphSAGE@64, @67, or GraphSAGE with the fixed 64+3 M2 input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The parent keeps the already-tested CLV coordinates identical. Its
        # NGCF parameters are removed before an optimizer can observe them.
        del self.sum_layers
        del self.bi_layers

        graph = self.adj.coalesce()
        n_nodes = self.n_users + self.n_items
        if graph.shape != (n_nodes, n_nodes):
            raise ValueError("인접행렬 shape이 전체 노드 수와 다릅니다")
        indices = graph.indices()
        receiver = indices[0]
        degree = torch.bincount(receiver, minlength=n_nodes)
        if (degree == 0).any():
            raise ValueError("GraphSAGE 평균전파에 고립 노드가 있습니다")
        values = degree[receiver].to(dtype=torch.float32).reciprocal()
        with torch.sparse.check_sparse_tensor_invariants(False):
            mean_adjacency = torch.sparse_coo_tensor(
                indices.clone(),
                values,
                graph.shape,
                device=indices.device,
            ).coalesce()
        del self.adj
        self.register_buffer("mean_adjacency", mean_adjacency, persistent=False)
        self.sage_layers = nn.ModuleList(
            GraphSAGEMeanLayer(self.input_dim) for _ in range(self.n_layers)
        )

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def propagation_layers(self) -> list[torch.Tensor]:
        user0, item0 = self.layer0_embeddings()
        current = torch.cat([user0, item0], dim=0)
        layers = [current]
        for layer in self.sage_layers:
            current = layer(current, self.mean_adjacency)
            layers.append(current)
        return layers

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        layers = self.propagation_layers()
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor | None) -> float:
            if parameter is None or parameter.grad is None:
                return 0.0
            return float(parameter.grad.norm())

        first = self.sage_layers[0].projection
        result = {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
            "graphsage_projection_layer0_gradient_norm": norm(first.weight),
            "graphsage_auxiliary_input_column_gradient_norm": (
                float(
                    torch.cat(
                        [
                            first.weight.grad[:, self.id_dim : self.input_dim],
                            first.weight.grad[
                                :, self.input_dim + self.id_dim :
                            ],
                        ],
                        dim=1,
                    ).norm()
                )
                if self.variant == "clv" and first.weight.grad is not None
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
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        result: dict[str, float | int | bool | str] = {
            "backbone": "GraphSAGE",
            "variant": self.variant,
            "id_dim": self.id_dim,
            "aux_dim": self.aux_dim,
            "layer_input_dim": self.input_dim,
            "final_embedding_dim": self.output_dim,
            "n_layers": self.n_layers,
            "neighbor_aggregation": "binary row mean",
            "self_neighbor_combination": "concat then learned projection",
            "layer_activation": "ELU",
            "layer_l2_normalization": True,
            "layer_aggregation": "mean(layer0,layer1,layer2)",
            "binary_graph": True,
            "one_dot_score": True,
            "external_reranking": False,
            "feature_dropout": 0.0,
        }
        row_sum = torch.sparse.sum(self.mean_adjacency, dim=1).to_dense()
        result["neighbor_mean_row_sum_max_error"] = float(
            (row_sum - 1.0).abs().max()
        )
        if self.variant == "clv":
            user_aux = self.user_auxiliary_coordinates()
            item_aux = self.item_auxiliary_coordinates()
            price_weights = torch.softmax(self.item_price_logits, dim=0)
            first = self.sage_layers[0].projection.weight
            first_aux = torch.cat(
                [
                    first[:, self.id_dim : self.input_dim],
                    first[:, self.input_dim + self.id_dim :],
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
                    (self.rho**0.5 * user_aux).norm(dim=1).mean()
                ),
                layer0_item_auxiliary_mean_norm=float(
                    (self.rho**0.5 * item_aux).norm(dim=1).mean()
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

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        keys = (
            "layer0_user_auxiliary_mean_norm",
            "layer0_item_auxiliary_mean_norm",
            "first_layer_auxiliary_column_norm",
            "item_price_weight_overall",
            "item_price_weight_within_category",
            "neighbor_mean_row_sum_max_error",
        )
        return {key: diagnostics[key] for key in keys if key in diagnostics}
