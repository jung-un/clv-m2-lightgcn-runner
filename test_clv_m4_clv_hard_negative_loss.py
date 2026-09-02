import pytest
import torch

from clv_m4_clv_hard_negative_loss import (
    clv_conditioned_negative_weights,
    multi_negative_bpr,
    sampled_l2_multineg,
)


def test_low_clv_uses_mean_and_high_clv_uses_highest_scored_negative():
    negative_scores = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]])
    q_clv = torch.tensor([0.0, 1.0])

    weights, hardest = clv_conditioned_negative_weights(
        negative_scores, q_clv
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0]]),
    )
    assert hardest.tolist() == [1, 0]


def test_intermediate_clv_is_exact_convex_mix_with_unit_row_mass():
    weights, _ = clv_conditioned_negative_weights(
        torch.tensor([[0.0, 1.0]]), torch.tensor([0.5])
    )

    torch.testing.assert_close(weights, torch.tensor([[0.25, 0.75]]))
    torch.testing.assert_close(weights.sum(1), torch.ones(1))


def test_diagnostics_report_effective_gradient_mass_not_only_nominal_weight_mass():
    positive = torch.tensor([1.0, 1.0])
    negatives = torch.tensor([[0.0, 0.9], [0.0, 0.9]])

    _, mean_diagnostics = multi_negative_bpr(
        positive, negatives, torch.tensor([0.0, 0.0])
    )
    _, hard_diagnostics = multi_negative_bpr(
        positive, negatives, torch.tensor([1.0, 1.0])
    )

    assert "effective_gradient_mass" in mean_diagnostics
    assert float(hard_diagnostics["effective_gradient_mass"]) > float(
        mean_diagnostics["effective_gradient_mass"]
    )


def test_k1_is_plain_bpr_for_every_clv_value():
    positive = torch.tensor([1.0, -0.5])
    negative = torch.tensor([[0.25], [0.5]])

    low, _ = multi_negative_bpr(positive, negative, torch.tensor([0.0, 0.0]))
    high, _ = multi_negative_bpr(positive, negative, torch.tensor([1.0, 1.0]))

    assert float(low) == pytest.approx(0.85006630, abs=1e-7)
    assert float(high) == pytest.approx(0.85006630, abs=1e-7)


def test_multineg_sampled_l2_averages_negative_rows_instead_of_summing_k():
    user = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    positive = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    negatives = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )

    value = sampled_l2_multineg(user, positive, negatives, coefficient=0.1)

    assert float(value) == pytest.approx(0.75, abs=1e-7)


@pytest.mark.parametrize(
    "scores,q_clv,message",
    [
        (torch.zeros(2, 0), torch.zeros(2), "K"),
        (torch.zeros(2, 3), torch.zeros(3), "shape"),
        (torch.zeros(2, 3), torch.tensor([-0.1, 0.5]), "범위"),
        (torch.zeros(2, 3), torch.tensor([0.5, float("nan")]), "유한"),
    ],
)
def test_weight_builder_rejects_invalid_inputs(scores, q_clv, message):
    with pytest.raises(ValueError, match=message):
        clv_conditioned_negative_weights(scores, q_clv)
