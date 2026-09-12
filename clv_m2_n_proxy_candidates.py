"""Train-history N candidates for the fixed q_V value-basis M2 screen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

from clv_dual_axis_model import fixed_percentile_ranks


@dataclass(frozen=True)
class BGNBDResult:
    expected_count: np.ndarray
    parameters: dict[str, float]
    diagnostics: dict[str, float | int | bool | str]


def recency_adjusted_activity(
    repeat_count: np.ndarray,
    transaction_recency: np.ndarray,
    customer_age: np.ndarray,
) -> np.ndarray:
    """Return (repeat count / age) times last-transaction freshness.

    ``transaction_recency`` is the elapsed time from a customer's first to
    most recent transaction and ``customer_age`` is the elapsed time from the
    first transaction to the calibration cutoff.  Their ratio is therefore
    near one when the most recent transaction is close to the cutoff.
    """

    x = np.asarray(repeat_count, dtype=np.float64)
    tx = np.asarray(transaction_recency, dtype=np.float64)
    age = np.asarray(customer_age, dtype=np.float64)
    if x.shape != tx.shape or x.shape != age.shape:
        raise ValueError("반복거래수·최신거래시점·관찰기간 shape이 다릅니다")
    if not all(np.isfinite(values).all() for values in (x, tx, age)):
        raise ValueError("최신성 보정 N 입력은 모두 유한해야 합니다")
    if np.any(x < 0.0) or np.any(age < 0.0):
        raise ValueError("반복거래수와 관찰기간은 음수일 수 없습니다")
    denominator = np.maximum(age, 1.0)
    freshness = np.clip(tx / denominator, 0.0, 1.0)
    return ((x / denominator) * freshness).astype(np.float32)


def _bgnbd_negative_log_likelihood(
    log_parameters: np.ndarray,
    frequency: np.ndarray,
    recency: np.ndarray,
    age: np.ndarray,
) -> float:
    """Mean negative BG/NBD log likelihood from Fader et al. (2005)."""

    r, alpha, a, b = np.exp(np.asarray(log_parameters, dtype=np.float64))
    if not np.isfinite([r, alpha, a, b]).all():
        return 1e100
    first = gammaln(r + frequency) - gammaln(r) + r * np.log(alpha)
    second = (
        gammaln(a + b)
        + gammaln(b + frequency)
        - gammaln(b)
        - gammaln(a + b + frequency)
    )
    active = -(r + frequency) * np.log(alpha + age)
    dropout = (
        np.log(a)
        - np.log(b + np.maximum(frequency, 1.0) - 1.0)
        - (r + frequency) * np.log(alpha + recency)
    )
    maximum = np.maximum(active, dropout)
    mixture = np.exp(active - maximum) + (frequency > 0.0) * np.exp(
        dropout - maximum
    )
    log_likelihood = first + second + np.log(mixture) + maximum
    value = -float(np.mean(log_likelihood))
    return value if np.isfinite(value) else 1e100


def _conditional_expected_count(
    horizon: float,
    frequency: np.ndarray,
    recency: np.ndarray,
    age: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    """Equation (10) conditional BG/NBD expected repeat transactions."""

    r = parameters["r"]
    alpha = parameters["alpha"]
    a = parameters["a"]
    b = parameters["b"]
    x = np.asarray(frequency, dtype=np.float64)
    tx = np.asarray(recency, dtype=np.float64)
    T = np.asarray(age, dtype=np.float64)
    z = horizon / (alpha + T + horizon)
    hyper_a = r + x
    hyper_b = b + x
    hyper_c = a + b + x - 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        direct = hyp2f1(hyper_a, hyper_b, hyper_c, z)
        alternate = hyp2f1(
            hyper_c - hyper_a,
            hyper_c - hyper_b,
            hyper_c,
            z,
        ) * np.power(1.0 - z, hyper_c - hyper_a - hyper_b)
        hyper = np.where(np.isfinite(direct) & (direct > 0.0), direct, alternate)
        exponent = np.log(hyper) + (r + x) * np.log(
            (alpha + T) / (alpha + T + horizon)
        )
        first = (a + b + x - 1.0) / (a - 1.0)
        numerator = first * (-np.expm1(exponent))
        alive_adjustment = np.ones_like(x)
        repeated = x > 0.0
        alive_adjustment[repeated] += (
            a
            / (b + x[repeated] - 1.0)
            * np.power(
                (alpha + T[repeated]) / (alpha + tx[repeated]),
                r + x[repeated],
            )
        )
        expected = numerator / alive_adjustment
    if not np.isfinite(expected).all():
        raise RuntimeError("BG/NBD 조건부 기대거래횟수 계산에 비유한 값이 있습니다")
    if float(expected.min(initial=0.0)) < -1e-7:
        raise RuntimeError("BG/NBD 조건부 기대거래횟수가 음수입니다")
    return np.maximum(expected, 0.0).astype(np.float32)


def fit_bgnbd_expected_count(
    repeat_count: np.ndarray,
    transaction_recency: np.ndarray,
    customer_age: np.ndarray,
    valid: np.ndarray,
    *,
    horizon: float = 7.0,
) -> BGNBDResult:
    """Fit one population BG/NBD and return each user's expected count.

    The likelihood and conditional expectation match the reference
    implementation in ``lifetimes.BetaGeoFitter``.  Frequency is the number
    of repeat transactions after the first purchase.
    """

    x = np.asarray(repeat_count, dtype=np.float64)
    tx = np.asarray(transaction_recency, dtype=np.float64)
    T = np.asarray(customer_age, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if x.shape != tx.shape or x.shape != T.shape or x.shape != valid.shape:
        raise ValueError("BG/NBD 입력 shape이 다릅니다")
    if horizon <= 0.0 or not np.isfinite(horizon):
        raise ValueError("BG/NBD 예측기간은 양수여야 합니다")
    usable = valid & np.isfinite(x) & np.isfinite(tx) & np.isfinite(T)
    usable &= (x >= 0.0) & (T >= 0.0) & (tx >= 0.0) & (tx <= T)
    if int(usable.sum()) < 20:
        raise ValueError("BG/NBD를 적합할 유효 사용자가 20명 미만입니다")

    fit_x = x[usable]
    fit_tx = tx[usable]
    fit_T = T[usable]
    scale = 1.0 / max(float(fit_T.max(initial=0.0)), 1.0)
    scaled_tx = fit_tx * scale
    scaled_T = fit_T * scale
    starts = (
        (0.5, 0.1, 1.5, 3.0),
        (1.0, 0.5, 0.8, 2.0),
        (0.2, 0.05, 2.0, 5.0),
    )
    fits = []
    for start in starts:
        fits.append(
            minimize(
                _bgnbd_negative_log_likelihood,
                np.log(np.asarray(start, dtype=np.float64)),
                args=(fit_x, scaled_tx, scaled_T),
                method="L-BFGS-B",
                bounds=[(-13.8, 13.8)] * 4,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
        )
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful:
        messages = "; ".join(str(fit.message) for fit in fits)
        raise RuntimeError(f"BG/NBD 최대우도 적합에 실패했습니다: {messages}")
    best = min(successful, key=lambda fit: float(fit.fun))
    r, scaled_alpha, a, b = np.exp(best.x)
    parameters = {
        "r": float(r),
        "alpha": float(scaled_alpha / scale),
        "a": float(a),
        "b": float(b),
    }
    expected = np.zeros_like(x, dtype=np.float32)
    expected[usable] = _conditional_expected_count(
        horizon,
        fit_x,
        fit_tx,
        fit_T,
        parameters,
    )
    diagnostics: dict[str, float | int | bool | str] = {
        "method": "BG/NBD maximum likelihood",
        "source": "Fader, Hardie, and Lee (2005), equation 10",
        "frequency_definition": "repeat transactions after first purchase",
        "horizon_days": float(horizon),
        "fit_user_count": int(usable.sum()),
        "fit_success": True,
        "optimizer_message": str(best.message),
        "negative_log_likelihood_mean": float(best.fun),
        "optimizer_iterations": int(best.nit),
        **parameters,
    }
    return BGNBDResult(expected, parameters, diagnostics)


def build_n_proxy_candidates(
    axes: dict,
    clv_valid: np.ndarray,
    *,
    horizon: float = 7.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build B/C/D raw N values and fixed percentile inputs."""

    required = (
        "valid_user",
        "activity_valid",
        "v_behavior_score",
        "repeat_transaction_count",
        "repeat_transaction_rate",
        "transaction_recency",
        "customer_age",
    )
    missing = [name for name in required if name not in axes]
    if missing:
        raise KeyError(f"N 후보 입력 누락: {missing}")
    valid_user = np.asarray(axes["valid_user"], dtype=bool)
    activity_valid = np.asarray(axes["activity_valid"], dtype=bool)
    clv_valid = np.asarray(clv_valid, dtype=bool)
    x = np.asarray(axes["repeat_transaction_count"], dtype=np.float64)
    tx = np.asarray(axes["transaction_recency"], dtype=np.float64)
    T = np.asarray(axes["customer_age"], dtype=np.float64)
    repeat_rate = np.asarray(axes["repeat_transaction_rate"], dtype=np.float64)
    v_raw = np.asarray(axes["v_behavior_score"], dtype=np.float64)
    if not all(values.shape == valid_user.shape for values in (clv_valid, x, tx, T, repeat_rate, v_raw)):
        raise ValueError("N 후보와 사용자 mask shape이 다릅니다")

    recency_activity = recency_adjusted_activity(x, tx, T)
    bgnbd = fit_bgnbd_expected_count(
        x,
        tx,
        T,
        valid_user & activity_valid,
        horizon=horizon,
    )
    raw = {
        "repeat_rate": repeat_rate.astype(np.float32),
        "recency_adjusted_activity": recency_activity,
        "bgnbd_expected_count": bgnbd.expected_count,
    }
    percentiles = {}
    rows = []
    for name, values in raw.items():
        q_n, _ = fixed_percentile_ranks(values, v_raw, valid_user)
        q_n = np.where(clv_valid, q_n, 0.0).astype(np.float32)
        percentiles[name] = q_n
        observed = values[clv_valid]
        rows.append(
            {
                "n_proxy": name,
                "valid_user_count": int(clv_valid.sum()),
                "raw_mean": float(observed.mean()) if len(observed) else 0.0,
                "raw_std": float(observed.std()) if len(observed) else 0.0,
                "raw_min": float(observed.min()) if len(observed) else 0.0,
                "raw_max": float(observed.max()) if len(observed) else 0.0,
                "q_n_mean": float(q_n[clv_valid].mean()) if clv_valid.any() else 0.0,
                "q_n_std": float(q_n[clv_valid].std()) if clv_valid.any() else 0.0,
            }
        )
    diagnostics = {
        "candidate_rows": rows,
        "bgnbd": bgnbd.diagnostics,
        "definitions": {
            "repeat_rate": "repeat_transaction_count / customer_age",
            "recency_adjusted_activity": (
                "repeat_rate * transaction_recency / customer_age"
            ),
            "bgnbd_expected_count": (
                "BG/NBD expected repeat transactions in the next 7 days"
            ),
        },
    }
    return percentiles, diagnostics
