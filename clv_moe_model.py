"""CLV-conditioned mixture of user-item embedding experts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_moe_features import ItemProfileArtifact, UserProfileArtifact


class EmbeddingExpert(nn.Module):
    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__()
        self.user = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.item = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.user[-1].weight, std=0.01)
        nn.init.normal_(self.item[-1].weight, std=0.01)
        nn.init.zeros_(self.user[-1].bias)
        nn.init.zeros_(self.item[-1].bias)


class CLVMixtureEmbeddingModel(nn.Module):
    """Mix expert embedding spaces with a user-level CLV behavior gate."""

    CONTROLS = {"clv", "constant_gate", "shuffled_clv", "single_adapter"}

    def __init__(
        self,
        base_model: nn.Module,
        user_profile: UserProfileArtifact,
        item_profile: ItemProfileArtifact,
        *,
        control: str = "clv",
        seed: int = 42,
        expert_count: int = 3,
        expert_hidden_dim: int = 32,
        expert_dim: int = 16,
        category_dim: int = 8,
    ):
        super().__init__()
        if control not in self.CONTROLS:
            raise ValueError(f"지원하지 않는 MoE control: {control}")
        if expert_count < 1:
            raise ValueError("expert_count는 1 이상이어야 합니다")
        self.base_model = base_model
        self.control = control
        requested_expert_count = int(expert_count)
        self.expert_count = 1 if control == "single_adapter" else requested_expert_count
        self.expert_dim = int(expert_dim)
        values = torch.as_tensor(user_profile.values, dtype=torch.float32)
        valid_user = torch.as_tensor(user_profile.valid_user, dtype=torch.bool)
        item_numeric = torch.as_tensor(item_profile.numeric, dtype=torch.float32)
        item_categories = torch.as_tensor(item_profile.category_ids, dtype=torch.long)
        valid_item = torch.as_tensor(item_profile.valid_item, dtype=torch.bool)
        self.register_buffer("original_profile", values.clone())
        self.register_buffer("has_profile", valid_user)
        self.register_buffer("item_numeric", item_numeric)
        self.register_buffer("item_category_ids", item_categories)
        self.register_buffer("valid_item", valid_item)

        routed = values.clone()
        if control == "shuffled_clv":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            valid_indices = torch.where(valid_user)[0]
            if len(valid_indices) > 1:
                permutation = valid_indices[
                    torch.randperm(len(valid_indices), generator=generator)
                ]
                routed[valid_indices] = values[permutation]
        self.register_buffer("routed_profile", routed)

        base_user, base_item, *_ = self.base_model.embeddings()
        user_input_dim = int(base_user.shape[1] + values.shape[1])
        item_input_dim = int(base_item.shape[1] + item_numeric.shape[1] + category_dim)
        category_parameters = int(item_profile.n_categories * category_dim)

        def expert_parameter_count(hidden: int) -> int:
            user_count = (user_input_dim + 1) * hidden + (hidden + 1) * expert_dim
            item_count = (item_input_dim + 1) * hidden + (hidden + 1) * expert_dim
            return user_count + item_count

        gate_parameters = (values.shape[1] + 1) * 32 + (32 + 1) * requested_expert_count
        target_parameters = (
            requested_expert_count
            * expert_parameter_count(expert_hidden_dim)
            + gate_parameters
            + category_parameters
        )
        selected_hidden = expert_hidden_dim
        if control == "single_adapter":
            selected_hidden = min(
                range(1, 513),
                key=lambda hidden: abs(
                    expert_parameter_count(hidden)
                    + category_parameters
                    - target_parameters
                ),
            )
            actual_parameters = (
                expert_parameter_count(selected_hidden) + category_parameters
            )
            self.parameter_match_ratio = actual_parameters / target_parameters
            if not 0.95 <= self.parameter_match_ratio <= 1.05:
                raise RuntimeError("single adapter 파라미터 수를 MoE와 5% 이내로 맞추지 못했습니다")
        else:
            self.parameter_match_ratio = 1.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.item_category = nn.Embedding(
                item_profile.n_categories, category_dim, padding_idx=0
            )
            self.experts = nn.ModuleList(
                [
                    EmbeddingExpert(
                        user_input_dim,
                        item_input_dim,
                        selected_hidden,
                        expert_dim,
                    )
                    for _ in range(self.expert_count)
                ]
            )
            self.gate_net = (
                None
                if control == "single_adapter"
                else nn.Sequential(
                    nn.Linear(values.shape[1], 32),
                    nn.GELU(),
                    nn.Linear(32, self.expert_count),
                )
            )

    def _base_embeddings(self):
        user, item, *_ = self.base_model.embeddings()
        return user, item

    def base_parameters(self) -> list[nn.Parameter]:
        return list(self.base_model.parameters())

    def adapter_parameters(self) -> list[nn.Parameter]:
        base_ids = {id(parameter) for parameter in self.base_model.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in base_ids]

    def routing_weights(self, users: torch.Tensor) -> torch.Tensor:
        if self.control == "single_adapter":
            return torch.ones(
                (len(users), 1), device=self.routed_profile.device, dtype=torch.float32
            )
        if self.control == "constant_gate":
            valid = self.has_profile
            if bool(valid.any()):
                mean = F.softmax(self.gate_net(self.routed_profile[valid]), dim=1).mean(
                    dim=0, keepdim=True
                )
            else:
                mean = torch.full(
                    (1, self.expert_count),
                    1.0 / self.expert_count,
                    device=self.routed_profile.device,
                )
            return mean.expand(len(users), -1)
        return F.softmax(self.gate_net(self.routed_profile[users]), dim=1)

    def user_expert_embeddings(self, users: torch.Tensor) -> torch.Tensor:
        base_user, _ = self._base_embeddings()
        inputs = torch.cat([base_user[users], self.routed_profile[users]], dim=1)
        return torch.stack([expert.user(inputs) for expert in self.experts], dim=1)

    def item_expert_embeddings(self, items: torch.Tensor | None = None) -> torch.Tensor:
        _, base_item = self._base_embeddings()
        if items is None:
            items = torch.arange(base_item.shape[0], device=base_item.device)
        category = self.item_category(self.item_category_ids[items])
        inputs = torch.cat(
            [base_item[items], self.item_numeric[items], category], dim=1
        )
        output = torch.stack([expert.item(inputs) for expert in self.experts], dim=1)
        return output * self.valid_item[items, None, None]

    def expert_embeddings(
        self, users: torch.Tensor, items: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.user_expert_embeddings(users),
            self.item_expert_embeddings(items),
            self.routing_weights(users),
        )

    def base_score_all(self, users: torch.Tensor) -> torch.Tensor:
        base_user, base_item = self._base_embeddings()
        return base_user[users] @ base_item.T

    def score_all(self, users: torch.Tensor, lam: float) -> torch.Tensor:
        base = self.base_score_all(users)
        if float(lam) == 0.0:
            return base
        user_experts, item_experts, gate = self.expert_embeddings(users)
        residual_score = torch.einsum(
            "uk,ukd,ikd->ui", gate, user_experts, item_experts
        )
        return base + float(lam) * self.has_profile[users, None] * residual_score

    def base_score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        base_user, base_item = self._base_embeddings()
        return (base_user[users] * base_item[items]).sum(dim=1)

    def score_pairs(
        self, users: torch.Tensor, items: torch.Tensor, lam: float
    ) -> torch.Tensor:
        base = self.base_score_pairs(users, items)
        if float(lam) == 0.0:
            return base
        user_experts = self.user_expert_embeddings(users)
        item_experts = self.item_expert_embeddings(items)
        gate = self.routing_weights(users)
        residual_score = (gate[:, :, None] * user_experts * item_experts).sum(
            dim=(1, 2)
        )
        return base + float(lam) * self.has_profile[users] * residual_score

    def bpr_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        lam: float = 1.0,
    ) -> torch.Tensor:
        positive_score = self.score_pairs(users, positives, lam)
        negative_score = self.score_pairs(users, negatives, lam)
        return -F.logsigmoid(positive_score - negative_score).mean()
