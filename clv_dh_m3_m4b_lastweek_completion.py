"""Complete the seed49 last-week 2x2 without retraining M1 or M5-B.

Only M3 and M4-B train. The previously exposed test is not used for tuning;
all four models are compared at the fixed 300-epoch endpoint, descriptively.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import clv_m2_m5_lastweek_test10 as common

M1, M3, M4_B, M5_B = common.M1, common.M3, common.M4_B, common.BASE
MODELS = (M1, M3, M4_B, M5_B)
REFERENCE_SOURCE = "6fb7f276c56e9c1888e2f448933556b4c78a46fb"
REFERENCE_RUN = "2483914fb871"
REFERENCE_SHA = "630ddcd9c531e55564f6a85ec1d93de24b6729500bfb1036536ba6d4513bc531"
REFERENCE_PATH = ("/content/drive/MyDrive/논문/data/"
                  "results_v3_dunnhumby_m2_m5_lastweek_seed49_pilot_v1/"
                  "2483914fb871/reports/result.json")


def configure(*, out_dir=None):
    return common.configure("dunnhumby", seeds=(49,),
                            experiment="dh_m3_m4b_completion", out_dir=out_dir)


def load_reference(path=REFERENCE_PATH):
    """Accept the exact original Drive JSON, or the user-supplied ZIP fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"기존 M1/M5-B 결과가 없습니다: {path}\n"
            "m2_m5_lastweek_dunnhumby_seed49_results.zip을 업로드하고 경로를 지정하세요. "
            "기준모형을 자동 재학습하지 않습니다.")
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            if archive.namelist().count("result.json") != 1:
                raise ValueError("ZIP의 result.json은 정확히 한 개여야 합니다")
            raw = archive.read("result.json")
    else:
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA:
        raise ValueError("승인된 seed49 마지막주 원본 result.json의 SHA256과 다릅니다")
    return json.loads(raw)


def validate_reference(reference, prep, cfg):
    old, current = reference["protocol"], prep["protocol"]
    expected_config = asdict(common.configure(seeds=(49,)))
    expected_config.pop("experiment")  # original source predates this selector
    expected_config.pop("out_dir")  # local and Colab roots differ, not a training setting
    old_settings = {k:v for k,v in old["config"].items() if k != "out_dir"}
    if old_settings != json.loads(json.dumps(expected_config)):
        raise ValueError("기존 실행 설정이 승인된 seed49 마지막주 실행과 다릅니다")
    actual_config = asdict(cfg)
    for key, value in old["config"].items():
        if key != "out_dir" and json.loads(json.dumps(actual_config[key])) != value:
            raise ValueError(f"기존 M1/M5-B와 설정 불일치: {key}")
    if (old["source_revision"] != REFERENCE_SOURCE
            or old["input_hash"] != prep["input_hash"]
            or old["intervals"] != [1, 704, 711]
            or old["validation"] or old["holdout"]):
        raise ValueError("기존 실행의 코드·입력·분할 불일치")
    for key in ("split", "intervals", "validation", "holdout", "data_stats",
                "no_early_stopping", "test_evaluations_per_fit", "test_checkpoint"):
        if old[key] != json.loads(json.dumps(current[key])):
            raise ValueError(f"기존 결과의 분할/평가 조건 불일치: {key}")
    if old["m4_weight_audit"]["sha256"] != current["m4_weight_audit"]["sha256"]:
        raise ValueError("기존 M5-B와 새 M4-B의 전체 학습행 가중치 SHA가 다릅니다")
    if not current["m4_weight_audit"]["invalid_extra_absent"]:
        raise ValueError("Dunnhumby 무효행 raw=1 규칙 불일치")
    for key in ("m3_beta",):
        if not np.isclose(old[key], current[key], rtol=1e-10, atol=1e-12):
            raise ValueError("기존 M5-B와 새 M3의 beta가 다릅니다")
    for key, value in old["m3_audit"].items():
        if not np.isclose(value, current["m3_audit"][key], rtol=1e-9, atol=1e-11):
            raise ValueError(f"기존 M5-B와 그래프 진단 불일치: {key}")
    selected = [r for r in reference["rows"] if r["model_id"] in (M1, M5_B)]
    if len(selected) != 2 or {r["model_id"] for r in selected} != {M1, M5_B}:
        raise ValueError("기존 M1/M5-B 결과 누락 또는 중복")
    metric_keys = set(selected[0]["metrics"])
    if len(metric_keys) != 142:
        raise ValueError("기존 전체142지표가 없습니다")
    for row in selected:
        identity = row["identity"]
        if (row["seed"] != 49 or row["epochs"] != 300
                or identity != dict(stage=common.CODE_VERSION, model_id=row["model_id"],
                                    seed=49, config_hash=REFERENCE_RUN,
                                    source_revision=REFERENCE_SOURCE, input_hash=prep["input_hash"])
                or [h["epoch"] for h in row["training_history"]] != list(range(1, 301))
                or set(row["metrics"]) != metric_keys
                or not np.isfinite(list(row["metrics"].values())).all()):
            raise ValueError("기존 결과의 신원·학습완료·전체지표 검증 실패")
    return selected


def prepare(cfg=None, *, reference_path=REFERENCE_PATH):
    cfg = cfg or configure()
    if cfg != configure(out_dir=cfg.out_dir):
        raise ValueError("Dunnhumby seed49·300epoch 고정설정을 변경할 수 없습니다")
    reference = load_reference(reference_path)  # fail before costly data preparation
    prep = common.prepare(cfg)
    prep["reference_rows"] = validate_reference(reference, prep, cfg)
    prep["reference_protocol"] = reference["protocol"]
    prep["reference_path"] = str(reference_path)
    prep["reference_sha256"] = REFERENCE_SHA
    common.io._atomic_json(prep["run_dir"] / "reference_audit.json", dict(
        result_sha256=REFERENCE_SHA, path=str(reference_path), source=REFERENCE_SOURCE,
        reference_run=REFERENCE_RUN, validation="input/config/split/population/weights/beta/metrics/history matched",
        retrain_references=False, reevaluate_references=False,
        reference_protocol=reference["protocol"]))
    print("[계획] M1·M5-B 원본 재사용 / M3 → M4-B만 각300epoch 학습 / seed49 고정", flush=True)
    return prep


def report(prep, cfg, rows):
    if (len(rows) != 4 or {r["model_id"] for r in rows} != set(MODELS)
            or any(r["seed"] != 49 or r["epochs"] != cfg.epochs for r in rows)):
        raise ValueError("네 모형의 고정점 결과가 모두 있어야 판독할 수 있습니다")
    metrics = list(rows[0]["metrics"])
    if any(set(r["metrics"]) != set(metrics) for r in rows):
        raise ValueError("모형 간 전체 평가 지표가 다릅니다")
    absolute = pd.DataFrame([dict(model_id=r["model_id"], seed=r["seed"],
                                 epoch=r["epochs"], **r["metrics"]) for r in rows])
    index = absolute.set_index("model_id").loc[list(MODELS)]
    if not np.isfinite(index[metrics].to_numpy(float)).all():
        raise ValueError("평가 지표 비유한값")
    comparisons = []
    for model, ref in ((M3,M1), (M4_B,M1), (M5_B,M1), (M5_B,M3), (M5_B,M4_B)):
        for metric in metrics:
            value, base = float(index.at[model,metric]), float(index.at[ref,metric])
            comparisons.append(dict(model_id=model, reference=ref, seed=49, epoch=cfg.epochs,
                                    metric=metric, value=value, reference_value=base, delta=value-base,
                                    relative_change_pct=100*(value/base-1) if base else None))
    interaction = pd.DataFrame([dict(
        metric=m, seed=49, epoch=cfg.epochs,
        m3_minus_m1=index.at[M3,m]-index.at[M1,m],
        m4_b_minus_m1=index.at[M4_B,m]-index.at[M1,m],
        m5_b_minus_m4_b=index.at[M5_B,m]-index.at[M4_B,m],
        interaction_absolute=index.at[M5_B,m]-index.at[M3,m]-index.at[M4_B,m]+index.at[M1,m]
    ) for m in metrics])
    guards = {model: bool(all(index.at[model,m] >= .99*index.at[M1,m]
                             for m in common.ACCURACY)) for model in (M3,M4_B,M5_B)}
    economic = bool(all(index.at[M5_B,m] > max(index.at[ref,m] for ref in (M1,M3,M4_B))
                        for m in common.ECONOMIC))
    single_guard = bool(all(index.at[M5_B,m] >= .99*max(index.at[M3,m],index.at[M4_B,m])
                            for m in common.ACCURACY))
    reading = dict(complete=True, seed_count=1, evaluated_epoch=cfg.epochs,
                   accuracy_guard_vs_m1=guards, m5_b_economic_above_m1_and_both_standalones=economic,
                   m5_b_accuracy_guard_vs_best_standalone=single_guard,
                   descriptive_combination_condition_met=bool(economic and guards[M5_B] and single_guard),
                   single_seed_diagnostic_only=True, previously_exposed_test=True,
                   significance_claim=False, clv_attribution_claim=False,
                   causal_interaction_claim=False, final_ten_seed_report=False,
                   no_post_test_selection=True)
    tables = dict(absolute=absolute, comparison=pd.DataFrame(comparisons), interaction=interaction,
                  diagnostics=pd.DataFrame([dict(model_id=r["model_id"], seed=49,
                                                 reused=r["model_id"] in (M1,M5_B),
                                                 **r["diagnostics"]) for r in rows]))
    root = prep["run_dir"] / "reports"
    paths = {k:str(root/f"{k}.csv") for k in tables}
    for key, table in tables.items():
        common.io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root/"result.json")
    common.io._atomic_json(Path(paths["json"]), dict(
        protocol=prep["protocol"], reference_protocol=prep["reference_protocol"],
        reference_sha256=prep["reference_sha256"], rows=rows, reading=reading, paths=paths))
    return dict(**tables, reading=reading, paths=paths)


def run(cfg, prep):
    if (cfg != configure(out_dir=cfg.out_dir)
            or json.loads(json.dumps(asdict(cfg))) != json.loads(json.dumps(prep["protocol"]["config"]))):
        raise ValueError("prepare 이후 설정이 변경됐습니다")
    # Re-read the pinned source and recheck the actual weight array before any fit.
    references = validate_reference(load_reference(prep["reference_path"]), prep, cfg)
    digest = hashlib.sha256(prep["m4_weights"].astype(np.float32).tobytes()).hexdigest()
    if digest != prep["protocol"]["m4_weight_audit"]["sha256"]:
        raise ValueError("prepare 이후 가중치가 변경됐습니다")
    rows = list(references)
    for model in (M3, M4_B):
        print(f"[추가학습] {model} / seed49 / 300epoch", flush=True)
        rows.append(common.run_arm(prep, cfg, model, 49))
    return report(prep, cfg, rows)
