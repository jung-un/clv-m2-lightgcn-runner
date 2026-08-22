"""M2 representation with fixed user gates applied after LightGCN propagation.

The binary graph and BPR samples are unchanged. ID, activity (N), and
transaction-value (V) blocks are learned in one optimizer. Unlike the
previous model, there is no learned dataset-wide N/V scalar. The propagated
N/V blocks are L2-normalized and only the fixed user-specific positive gates
control their final contribution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clv_joint_nv_model import JointNVLightGCN


class PostPropagationGatedJointNVLightGCN(JointNVLightGCN):
    """One LightGCN with ID/N/V blocks and post-propagation user gates."""

    def __init__(self, **kwargs):
        # The parent validates this placeholder; learned global scalars are
        # removed immediately and never enter the model or optimizer.
        kwargs = {**kwargs, "gamma_init": 0.1}
        super().__init__(**kwargs)
        del self.sqrt_gamma_n
        del self.sqrt_gamma_v

    @property
    def gamma_n(self) -> torch.Tensor:
        return self.E_u.weight.new_ones(())

    @property
    def gamma_v(self) -> torch.Tensor:
        return self.E_u.weight.new_ones(())

    @property
    def activity_axis_weight(self) -> None:
        return None

    @property
    def transaction_value_axis_weight(self) -> None:
        return None

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ID/N/V blocks without global weights or user gates."""
        user_n = self.activity_user(self.user_activity)
        user_v = self.value_user(self.user_value)
        item_n = self.activity_item(self.item_activity) * self.valid_item[:, None]
        item_v = self.value_item(self.item_value) * self.valid_item[:, None]
        user = torch.cat([self.E_u.weight, user_n, user_v], dim=1)
        item = torch.cat([self.E_i.weight, item_n, item_v], dim=1)
        return user, item

    @staticmethod
    def _normalize(block: torch.Tensor) -> torch.Tensor:
        return F.normalize(block, dim=1, eps=1e-8)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate once, then normalize N/V and allocate them by user."""
        user0, item0 = self.layer0_embeddings()
        current = torch.cat([user0, item0], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        user = total[: self.n_users]
        item = total[self.n_users :]

        user_id = user[:, : self.id_dim]
        user_n = self._normalize(
            user[:, self.id_dim : self.id_dim + self.axis_dim]
        )
        user_v = self._normalize(user[:, self.id_dim + self.axis_dim :])
        item_id = item[:, : self.id_dim]
        item_n = self._normalize(
            item[:, self.id_dim : self.id_dim + self.axis_dim]
        )
        item_v = self._normalize(item[:, self.id_dim + self.axis_dim :])

        # Invalid users stay zero even if item messages reached them.
        user_n = (
            user_n
            * self.user_activity_valid[:, None]
            * self.gate_n[:, None]
        )
        user_v = (
            user_v
            * self.user_value_valid[:, None]
            * self.gate_v[:, None]
        )
        item_n = item_n * self.valid_item[:, None]
        item_v = item_v * self.valid_item[:, None]
        return (
            torch.cat([user_id, user_n, user_v], dim=1),
            torch.cat([item_id, item_n, item_v], dim=1),
        )

    @torch.no_grad()
    def score_diagnostics(self, n_sample: int = 512, seed: int = 0) -> dict:
        diagnostics = super().score_diagnostics(n_sample=n_sample, seed=seed)
        diagnostics.update(
            learned_global_axis_weights=False,
            gate_application="after_propagation",
            axis_normalization="l2_after_propagation",
        )
        return diagnostics
