"""No-training check behind M5-B's N weight: does q_N pick the customers who get evaluated?

M5-B multiplies every train row of customer u by (1 + .5*q_N(u)).  That only
makes sense if q_N identifies who will actually show up (and be scored) in the
evaluation week.  For each dataset this reports, over training customers:

* Spearman of q_N / q_V / q_C / binary degree with "is an evaluation customer"
  (bought at least one new-to-user item in the development week);
* the same for q_N inside degree deciles, because rows already scale with
  degree in BPR — q_N must add something degree does not.

Development split only (Dunnhumby 684-690, H&M 2020-09-02..08); no model, no test.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def summarise(q_n, q_v, q_c, valid, degree, eval_users) -> dict:
    n = len(q_n)
    is_eval = np.zeros(n, bool)
    is_eval[np.asarray(eval_users, np.int64)] = True
    keep = valid & (degree > 0)
    frame = pd.DataFrame({"q_n": q_n, "q_v": q_v, "q_c": q_c, "degree": degree,
                          "eval": is_eval.astype(float)})[keep]
    out = {"customers": int(keep.sum()), "eval_customers": int(frame["eval"].sum()),
           "eval_rate": float(frame["eval"].mean())}
    for col in ("q_n", "q_v", "q_c", "degree"):
        out[f"spearman_{col}_vs_eval"] = float(spearmanr(frame[col], frame["eval"]).statistic)
    decile = pd.qcut(frame["degree"].rank(method="first"), 10, labels=False)
    within = [spearmanr(g.q_n, g["eval"]).statistic for _, g in frame.groupby(decile)
              if g["eval"].nunique() > 1]
    out["q_n_vs_eval_within_degree_decile_mean"] = float(np.mean(within))
    out["q_n_vs_eval_within_degree_decile_positive"] = f"{sum(r > 0 for r in within)}/{len(within)}"
    out["eval_rate_by_q_n_decile"] = [round(float(x), 4) for x in
        frame.groupby(pd.qcut(frame.q_n.rank(method="first"), 10, labels=False))["eval"].mean()]
    return out


def self_test() -> None:
    rng = np.random.default_rng(0)
    q_n = rng.random(2000)
    degree = rng.integers(1, 50, 2000)
    eval_users = np.flatnonzero(rng.random(2000) < q_n)  # eval probability rises with q_n
    r = summarise(q_n, rng.random(2000), q_n, np.ones(2000, bool), degree, eval_users)
    assert r["spearman_q_n_vs_eval"] > 0.3 and r["q_n_vs_eval_within_degree_decile_mean"] > 0.3
    assert abs(r["spearman_q_v_vs_eval"]) < 0.1


def dunnhumby() -> dict:
    import lightgcn_clv_component_recheck as recheck
    p = recheck._prepare(recheck.configure_component_recheck())
    d = p["data"]
    pairs = np.unique(np.asarray(d["pos_key"], np.int64))  # distinct (user, item)
    degree = np.bincount(pairs // d["n_items"], minlength=d["n_users"])
    return summarise(np.asarray(p["q_n"]), np.asarray(p["q_v"]), np.asarray(p["q_c"]),
                     np.asarray(p["clv_valid"], bool), degree, p["cache"].users)


def hm() -> dict:
    import lightgcn_clv_hm2y_seed42_common as common
    import lightgcn_clv_m3_centered_value_graph_hm2y as hm2y
    cfg = hm2y.configure_centered_graph_hm2y(
        out_dir=str(Path(tempfile.gettempdir()) / "clv_m5_qn_precheck_hm"))
    p = common.prepare_hm2y(cfg, code_version="clv-m5-qn-eval-user-precheck-v1")
    return summarise(np.asarray(p["q_n"]), np.asarray(p["q_v"]), np.asarray(p["q_c"]),
                     np.asarray(p["clv_valid"], bool), p["binary_user_degree"],
                     p["cache"].users)


if __name__ == "__main__":
    self_test()
    which = sys.argv[1:] or ["dunnhumby", "hm"]
    print(json.dumps({name: globals()[name]() for name in which}, ensure_ascii=False, indent=1))
