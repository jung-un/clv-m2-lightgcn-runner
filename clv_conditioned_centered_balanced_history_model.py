"""Centered, balanced CLV-conditioned category/price history LightGCN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_conditioned_category_price_history_model import (
    ConditionedCategoryPriceHistoryLightGCN,
)


class CenteredBalancedHistoryLightGCN(ConditionedCategoryPriceHistoryLightGCN):
    """Center and balance the two auxiliary relations before propagation."""

    def __init__(self, *args, warmup_epochs: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        if warmup_epochs <= 0:
            raise ValueError("warmup_epochs는 양수여야 합니다")
        self.rho_max = float(self.rho)
        self.warmup_epochs = int(warmup_epochs)
        self.condition_mixer = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.condition_mixer.weight)

    def set_training_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("epoch는 1 이상이어야 합니다")
        self.rho = self.rho_max * min(1.0, float(epoch) / self.warmup_epochs)

    def _gate(self) -> torch.Tensor:
        category_weight = 0.25 + 0.5 * torch.sigmoid(
            self.condition_mixer(self.user_state)
        )
        return torch.cat([category_weight, 1.0 - category_weight], dim=1)

    def _raw_histories(self, category: torch.Tensor):
        return (
            torch.sparse.mm(self.category_history, category),
            torch.sparse.mm(self.price_history, category),
        )

    def _population_statistics(
        self, history_category: torch.Tensor, history_price: torch.Tensor
    ):
        valid = self.auxiliary_valid[:, None]
        count = valid.sum().clamp_min(1.0)
        return (
            (history_category * valid).sum(0),
            (history_price * valid).sum(0),
            count,
        )

    @staticmethod
    def _center_and_unit(
        values: torch.Tensor,
        mean: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize((values - mean) * valid, dim=1, eps=1e-8) * valid

    def _layer0_blocks(self):
        category = self._unit_rows(self.category_embedding.weight)
        raw_category, raw_price = self._raw_histories(category)
        category_sum, price_sum, valid_count = self._population_statistics(
            raw_category, raw_price
        )
        valid = self.auxiliary_valid[:, None]
        history_category = self._center_and_unit(
            raw_category, category_sum / valid_count, valid
        )
        history_price = self._center_and_unit(
            raw_price, price_sum / valid_count, valid
        )
        gate = self._gate()
        user_aux = torch.cat(
            [gate[:, :1] * history_category, gate[:, 1:] * history_price], dim=1
        )
        item_category = category[self.item_category]
        item_price = self.item_price_signal[:, None] * item_category
        item_aux = torch.cat([item_category, item_price], dim=1)
        return user_aux, item_aux, history_category, history_price, category, gate

    def _leave_one_out_auxiliary(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        hcat: torch.Tensor,
        hprice: torch.Tensor,
        category: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        del hcat, hprice
        raw_category, raw_price = self._raw_histories(category)
        category_sum, price_sum, valid_count = self._population_statistics(
            raw_category, raw_price
        )
        count = self.unique_item_count[users]
        positive_category = self.item_category[positives]
        positive_category_vector = category[positive_category]
        positive_price_vector = (
            self.item_price_signal[positives, None] * positive_category_vector
        )
        remaining = (count - 1.0).clamp_min(1.0)
        loo_category_raw = (
            count[:, None] * raw_category[users] - positive_category_vector
        ) / remaining[:, None]
        loo_price_raw = (
            count[:, None] * raw_price[users] - positive_price_vector
        ) / remaining[:, None]
        eligible = (
            (count > 1.0).float()[:, None] * self.auxiliary_valid[users, None]
        )
        category_mean = (
            category_sum[None, :]
            + eligible * (loo_category_raw - raw_category[users])
        ) / valid_count
        price_mean = (
            price_sum[None, :] + eligible * (loo_price_raw - raw_price[users])
        ) / valid_count
        loo_category = self._center_and_unit(
            loo_category_raw, category_mean, eligible
        )
        loo_price = self._center_and_unit(loo_price_raw, price_mean, eligible)
        return torch.cat(
            [
                gate[users, :1] * loo_category,
                gate[users, 1:] * loo_price,
            ],
            dim=1,
        )

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float]:
        user, item, hcat, hprice, category, gate = self._propagated_blocks()
        valid = self.auxiliary_valid.bool()
        alpha = gate[valid, 0]
        beta = gate[valid, 1]
        user_id, item_id = self.id_only_embeddings()
        user_aux = user[:, self.id_dim :]
        item_aux = item[:, self.id_dim :]
        sample_users = torch.where(valid)[0][: min(512, int(valid.sum()))]
        sample_items = torch.arange(
            min(2048, self.n_items), device=item.device, dtype=torch.long
        )
        id_scores = user_id[sample_users] @ item_id[sample_items].T
        aux_scores = user_aux[sample_users] @ item_aux[sample_items].T
        normalized_aux = F.normalize(user_aux[valid], dim=1, eps=1e-8)
        n_valid_aux = len(normalized_aux)
        if n_valid_aux > 1:
            cosine_sum = normalized_aux.sum(0).pow(2).sum() - n_valid_aux
            mean_pair_cosine = cosine_sum / (n_valid_aux * (n_valid_aux - 1))
        else:
            mean_pair_cosine = user_aux.new_zeros(())
        raw_category, raw_price = self._raw_histories(category)
        category_sum, price_sum, valid_count = self._population_statistics(
            raw_category, raw_price
        )
        raw_valid = self.auxiliary_valid[:, None]
        centered_category = (raw_category - category_sum / valid_count) * raw_valid
        centered_price = (raw_price - price_sum / valid_count) * raw_valid
        state = self.user_state[valid]
        result = {
            "rho": self.rho,
            "rho_max": self.rho_max,
            "warmup_epochs": self.warmup_epochs,
            "total_dim": self.total_dim,
            "alpha_mean": float(alpha.mean()) if len(alpha) else 0.0,
            "alpha_std": float(alpha.std()) if len(alpha) else 0.0,
            "alpha_min": float(alpha.min()) if len(alpha) else 0.0,
            "alpha_max": float(alpha.max()) if len(alpha) else 0.0,
            "beta_mean": float(beta.mean()) if len(beta) else 0.0,
            "beta_std": float(beta.std()) if len(beta) else 0.0,
            "auxiliary_score_std_ratio_to_id": float(
                aux_scores.std() / id_scores.std().clamp_min(1e-8)
            ),
            "category_history_mean_norm": float(hcat[valid].norm(dim=1).mean())
            if valid.any()
            else 0.0,
            "price_history_mean_norm": float(hprice[valid].norm(dim=1).mean())
            if valid.any()
            else 0.0,
            "category_centered_population_mean_norm": float(
                centered_category[valid].mean(0).norm()
            )
            if valid.any()
            else 0.0,
            "price_centered_population_mean_norm": float(
                centered_price[valid].mean(0).norm()
            )
            if valid.any()
            else 0.0,
            "user_auxiliary_mean_pair_cosine": float(mean_pair_cosine),
            "condition_mixer_weight_norm": float(self.condition_mixer.weight.norm()),
            "condition_mixer_bias_norm": 0.0,
        }
        names = ("n_hat", "v_hat", "clv_proxy", "n_minus_v")
        for column, name in enumerate(names):
            result[f"alpha_{name}_correlation"] = self._corr(alpha, state[:, column])
        return self.feature_diagnostics | result

