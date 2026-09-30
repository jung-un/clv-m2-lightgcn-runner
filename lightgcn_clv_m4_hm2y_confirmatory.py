"""Score the finished H&M M4 arms once on the protected test split. No training.

The three-arm H&M screen passed its pre-registered attribution rule on seeds 42
and 43, but every number so far comes from the development split
(validation, 2020-09-02~08), which the project has looked at repeatedly. The
test split (2020-09-09~15) has never been computed: the development chain pins
``EVAL_TEST=False`` and validates it at three layers.

This runner opens it once. It trains nothing - it loads each arm's finished
checkpoint, re-scores it on validation to prove the weights and inputs are the
ones that produced the reported development numbers, and only then scores the
test split.

Opening the test split is one-way. Everything reported is fixed here before the
run: the arms, the seeds, the metrics and the four conditions are copied from
the frozen development screen, and no epoch, arm or seed is chosen afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_m4_k1_assignment_control_hm2y as screen
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_k1_m4_improvement_screen as training
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m4-hm2y-confirmatory-test-v1"
DEV_SPLIT = screen.SPLIT_LABEL
TEST_SPLIT = "hm2y_test_2020-09-09_15"
ECONOMIC_METRICS = screen.ECONOMIC_METRICS
ACCURACY_METRICS = screen.ACCURACY_METRICS
ACCURACY_GUARD = screen.ACCURACY_GUARD
# 개발 결과를 재현하는지 확인할 때 허용하는 오차. 같은 가중치·같은 입력이면
# 완전히 같은 수가 나와야 하므로 수치오차 수준만 허용한다.
REPRODUCTION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ConfirmatoryConfig:
    seeds: tuple[int, ...] = (42, 43)
    open_test_split: bool = False      # 명시적으로 True를 줘야 보호분할을 연다
    out_dir: str = ""
    dev_out_dir: str = ""


def configure_confirmatory(**overrides) -> ConfirmatoryConfig:
    root = v3.default_out_dir("hm")
    defaults = {
        "out_dir": f"{root}_m4_k1_assignment_control_hm2y_confirmatory_v1",
        "dev_out_dir": (
            f"{root}_m4_k1_assignment_control_hm2y_development_screen_v1"
        ),
    }
    return ConfirmatoryConfig(**(defaults | overrides))


def preflight_summary(cfg: ConfirmatoryConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "trains_nothing": True,
        "development_split": DEV_SPLIT,
        "protected_split_opened": TEST_SPLIT,
        "seeds": list(cfg.seeds),
        "arms": list(screen.MODEL_IDS),
        "primary_metrics": list(ECONOMIC_METRICS),
        "conditions_copied_from_the_frozen_screen": {
            "actual_beats_baseline": "actual M4 > M1 on both primary metrics",
            "actual_beats_shuffle": "actual M4 > q_C shuffle on both primary metrics",
            "accuracy_guard": f"six accuracy metrics >= {ACCURACY_GUARD} of M1",
        },
        "verification_before_opening_the_test_split": (
            "every arm is re-scored on validation and must reproduce its reported "
            f"development metrics to {REPRODUCTION_TOLERANCE}"
        ),
        "nothing_selected_afterwards": (
            "arms, seeds, metrics and conditions are fixed above; no epoch, arm or "
            "seed is chosen after seeing the test numbers"
        ),
        "limits": (
            "two seeds, no confidence intervals, one evaluation of one protected "
            "split; the holdout stays closed"
        ),
    }


# --------------------------------------------------------------------------
# loading finished arms
# --------------------------------------------------------------------------


def discover_arm_hash(dev_out_dir: str, seed: int) -> str:
    """Find the folder the finished arms actually sit in.

    The development config hash includes the source revision, so recomputing it
    here - from a later commit that carries this runner - points at a folder that
    was never written. The arms are located by their file names instead, and all
    three must live in the same folder for the seed to be usable.
    """

    root = Path(dev_out_dir) / "arms"
    folders = set()
    for model_id in screen.MODEL_IDS:
        found = sorted(root.glob(f"*/{model_id}_s{seed}.json"))
        if len(found) != 1:
            raise RuntimeError(
                f"seed {seed} {model_id}의 결과 파일이 {len(found)}개입니다 "
                f"({[str(f) for f in found]}). 이 실행은 학습하지 않습니다."
            )
        folders.add(found[0].parent.name)
    if len(folders) != 1:
        raise RuntimeError(f"seed {seed}의 arm이 서로 다른 폴더에 있습니다: {sorted(folders)}")
    return folders.pop()


def _arm_config(seed: int, dev_out_dir: str):
    return screen.configure_hm2y_m4_assignment_screen(
        seed=seed, shuffle_seed=seed, out_dir=dev_out_dir
    )


def load_finished_arms(seed: int, cfg: ConfirmatoryConfig) -> tuple[dict, dict]:
    """Return the prepared development inputs and every arm's finished model.

    Refuses to continue when a checkpoint is missing: the development runner
    would start training at that point, and this run must not train.
    """

    arm_cfg = _arm_config(seed, cfg.dev_out_dir)
    prepared = screen._prepare(arm_cfg)
    recomputed = prepared["config_hash"]
    prepared["config_hash"] = discover_arm_hash(cfg.dev_out_dir, seed)
    print(f"  arm 폴더 {prepared['config_hash']} (이 커밋에서 다시 계산하면 {recomputed} — "
          f"설정 해시에 소스 커밋이 들어가므로 다릅니다)")
    models: dict[str, object] = {}
    payloads: dict[str, dict] = {}
    for spec in screen.arm_specifications(prepared):
        paths = legacy._arm_paths(prepared, arm_cfg, spec["model_id"])
        for name in ("result", "checkpoint"):
            if not paths[name].is_file():
                raise RuntimeError(
                    f"seed {seed} {spec['model_id']}의 {name} 파일이 없습니다: "
                    f"{paths[name]}. 이 실행은 학습하지 않습니다."
                )
        payload, model = training._run_arm(
            screen.arm_prepared(prepared, spec), arm_cfg, spec
        )
        model.eval()
        models[spec["model_id"]] = model
        payloads[spec["model_id"]] = payload
    return prepared, {"models": models, "payloads": payloads, "cfg": arm_cfg}


@torch.no_grad()
def evaluate_on(model, prepared: dict, cache) -> dict:
    """Score one model on one evaluation cache with the development settings.

    Goes through the same flattening the development runner used, so the keys
    here are the reported ones (`recall@10`) rather than the nested per-cut-off
    structure `v3.evaluate` returns.
    """

    flat, _ = moe._flat_evaluation(
        model, 0.0, cache, prepared["meta"], prepared["data"],
        prepared["base_cfg"], per_user=False,
    )
    return test10._public_metrics(flat)


def verify_development_reproduction(prepared: dict, arms: dict) -> pd.DataFrame:
    """Re-score every arm on validation; the numbers must match what was reported.

    This is what makes opening the test split safe. If the weights or the inputs
    were not the ones behind the development result, the validation numbers would
    differ and the run stops before the protected split is touched.
    """

    rows = []
    wanted = list(ACCURACY_METRICS) + list(ECONOMIC_METRICS)
    for model_id, model in arms["models"].items():
        reported = arms["payloads"][model_id]["metrics"]
        rescored = evaluate_on(model, prepared, prepared["cache"])
        for name, source in (("재채점", rescored), ("보고된 개발", reported)):
            missing = [m for m in wanted if m not in source]
            if missing:
                raise RuntimeError(
                    f"{model_id}의 {name} 지표에 {missing}가 없습니다. "
                    f"가진 키 예: {sorted(source)[:6]}"
                )
        for metric in wanted:
            gap = abs(float(rescored[metric]) - float(reported[metric]))
            rows.append({"model_id": model_id, "metric": metric,
                         "reported": float(reported[metric]),
                         "rescored": float(rescored[metric]), "abs_gap": gap})
    frame = pd.DataFrame(rows)
    worst = frame.abs_gap.max()
    if worst > REPRODUCTION_TOLERANCE:
        offender = frame.loc[frame.abs_gap.idxmax()]
        raise RuntimeError(
            f"개발 결과가 재현되지 않습니다 (최대 오차 {worst:.3e}, "
            f"{offender.model_id}/{offender.metric}). 보호분할을 열지 않습니다."
        )
    print(f"  개발 결과 재현 확인: 최대 오차 {worst:.3e}")
    return frame


# --------------------------------------------------------------------------
# the protected split
# --------------------------------------------------------------------------


def build_test_cache(prepared: dict):
    """Build the test-split evaluation cache without disturbing the training data.

    The development chain fixes EVAL_TEST=False and checks it, so the test split
    is constructed here from a separate run configuration. Training stays at
    train <= 2020-09-01, so the customer and item indices are the development
    ones; that is asserted rather than assumed.
    """

    data = prepared["data"]
    base = dict(prepared["base_cfg"])
    configured = v3.configure_run(
        "hm", out_dir=base["OUT_DIR"], EVAL_TEST=True, EVAL_HOLDOUT=False,
        TRAIN_ON_VAL=False,
    )
    opened = v3.prepare_data(configured, v3.DCFG)
    for key in ("n_users", "n_items"):
        if opened[key] != data[key]:
            raise RuntimeError(
                f"보호분할 준비에서 {key}가 달라졌습니다: {opened[key]} != {data[key]}"
            )
    if len(opened["tr_u"]) != len(data["tr_u"]):
        raise RuntimeError("학습 행 수가 달라졌습니다 — 같은 학습 구간이 아닙니다")
    if "test" not in opened["splits"]:
        raise RuntimeError("test 분할이 만들어지지 않았습니다")
    cache = v3.EvalCache(
        *opened["splits"]["test"], prepared["axes"]["clv_proxy"],
        v3.segment_thresholds(prepared["axes"]["clv_proxy"], base["SEG_EDGES"]),
        data["n_items"],
    )
    print(f"  보호분할 평가유저 {len(cache.users):,}명")
    return cache


def confirmatory_reading(rows: pd.DataFrame) -> dict:
    """The frozen screen's conditions, applied to the protected split."""

    reading = {"split": TEST_SPLIT, "seeds": sorted(rows.seed.unique().tolist()),
               "significance_claimed": False, "selected_after_seeing": False}
    per_seed = {}
    for seed, part in rows.groupby("seed"):
        table = part.set_index("model_id")
        m1 = table.loc[screen.M1_MODEL_ID]
        actual = table.loc[screen.M4_ACTUAL_MODEL_ID]
        shuffled = table.loc[screen.M4_SHUFFLED_MODEL_ID]
        beats_m1 = all(actual[m] > m1[m] for m in ECONOMIC_METRICS)
        beats_shuffle = all(actual[m] > shuffled[m] for m in ECONOMIC_METRICS)
        guard = all(actual[m] >= ACCURACY_GUARD * m1[m] for m in ACCURACY_METRICS)
        per_seed[int(seed)] = {
            "actual_beats_m1_on_both_economic_metrics": bool(beats_m1),
            "actual_beats_shuffle_on_both_economic_metrics": bool(beats_shuffle),
            "six_accuracy_metrics_at_least_99pct_of_m1": bool(guard),
            "attribution_pass": bool(beats_m1 and beats_shuffle and guard),
            "worst_accuracy_ratio_vs_m1": float(
                min(actual[m] / m1[m] for m in ACCURACY_METRICS)
            ),
            "deltas_actual_minus_m1": {
                m: float(actual[m] - m1[m])
                for m in list(ACCURACY_METRICS) + list(ECONOMIC_METRICS)
            },
            "deltas_actual_minus_shuffle": {
                m: float(actual[m] - shuffled[m])
                for m in list(ACCURACY_METRICS) + list(ECONOMIC_METRICS)
            },
        }
    reading["per_seed"] = per_seed
    reading["seeds_passing"] = sum(v["attribution_pass"] for v in per_seed.values())
    reading["seeds_evaluated"] = len(per_seed)
    return reading


def run_confirmatory(cfg: ConfirmatoryConfig | None = None) -> pd.DataFrame:
    cfg = cfg or configure_confirmatory()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    if not cfg.open_test_split:
        raise RuntimeError(
            "보호분할은 한 번만 열 수 있습니다. 보고할 내용을 확정한 뒤 "
            "open_test_split=True로 실행하세요."
        )

    rows, checks = [], []
    for seed in cfg.seeds:
        print(f"\n===== seed {seed} =====")
        prepared, arms = load_finished_arms(seed, cfg)
        checks.append(verify_development_reproduction(prepared, arms).assign(seed=seed))
        cache = build_test_cache(prepared)
        for model_id, model in arms["models"].items():
            metrics = evaluate_on(model, prepared, cache)
            rows.append({"seed": seed, "model_id": model_id, "split": TEST_SPLIT,
                         **metrics})
            print(f"  {model_id}: recall@10 {metrics['recall@10']:.6f} | "
                  f"가중 적중값@10 {metrics[ECONOMIC_METRICS[0]]:.6f}")

    frame = pd.DataFrame(rows)
    verification = pd.concat(checks, ignore_index=True)
    reading = confirmatory_reading(frame)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"clv_m4_hm2y_confirmatory_{'_'.join(str(s) for s in cfg.seeds)}"
    paths = {"test_csv": out / f"{stem}_test.csv",
             "verification_csv": out / f"{stem}_verification.csv",
             "json": out / f"{stem}.json"}
    test10._atomic_csv(paths["test_csv"], frame)
    test10._atomic_csv(paths["verification_csv"], verification)
    test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_rows": frame.to_dict("records"), "reading": reading,
        "result_paths": {k: str(v) for k, v in paths.items()},
    })
    print("\n확증 판독:", json.dumps(reading, ensure_ascii=False, indent=2))
    print("저장:", json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))
    frame.attrs.update(reading=reading, result_paths={k: str(v) for k, v in paths.items()})
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_confirmatory()),
                     ensure_ascii=False, indent=2))
