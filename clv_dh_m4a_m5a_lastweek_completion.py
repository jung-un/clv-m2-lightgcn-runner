"""Add original M4-A and M5-A to the pinned Dunnhumby last-week seed49 table.

The existing M1/M3/M4-B/M5-B rows are reused byte-for-byte from one SHA-pinned
result.  M4-A and M5-A each start from random initialisation and train for the
fixed 300 epochs with one optimizer.  This is an exposed-test, one-seed pilot.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import clv_m2_m5_lastweek_test10 as common

M1, M3 = common.M1, common.M3
M4_A, M5_A = common.M4_A, common.M5_A
M4_B, M5_B = common.M4_B, common.BASE
MODELS = (M1, M3, M4_A, M5_A, M4_B, M5_B)
REFERENCE_SHA = "f0916d978e78cbea017b3f3739cb13f57ee573aae698bdd48a4e1a9ab0688782"
REFERENCE_PATH = ("/content/drive/MyDrive/논문/data/"
                  "results_v3_dunnhumby_m3_m4b_lastweek_seed49_v1/"
                  "1c588fd8e51d/reports/result.json")


def configure(*, out_dir=None):
    return common.configure(
        "dunnhumby", seeds=(49,), experiment="dh_m4a_m5a_completion",
        out_dir=out_dir)


def load_reference(path=REFERENCE_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"기존 4모형 결과가 없습니다: {path}\n"
            "dh_m3_m4b_lastweek_seed49_results.zip을 업로드하고 경로를 지정하세요. "
            "기준모형을 자동 재학습하지 않습니다.")
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith("result.json")]
            if len(names) != 1:
                raise ValueError("ZIP의 result.json은 정확히 한 개여야 합니다")
            raw = archive.read(names[0])
    else:
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA:
        raise ValueError("승인된 seed49 4모형 result.json의 SHA256과 다릅니다")
    return json.loads(raw)


def validate_reference(reference, prep, cfg):
    old, current = reference["protocol"], prep["protocol"]
    fixed = ("dataset", "seeds", "epochs", "id_dim", "n_layers", "batch_size",
             "lr", "pref_reg", "negative_count", "input_days", "basis_bandwidth",
             "target_cv", "max_degree_correlation", "max_price_correlation",
             "max_popularity_correlation")
    for key in fixed:
        if (json.loads(json.dumps(old["config"][key]))
                != json.loads(json.dumps(getattr(cfg, key)))):
            raise ValueError(f"기존 4모형과 학습 설정 불일치: {key}")
    for key in ("split", "intervals", "validation", "holdout", "data_stats",
                "no_early_stopping", "test_evaluations_per_fit", "test_checkpoint"):
        if old[key] != json.loads(json.dumps(current[key])):
            raise ValueError(f"기존 4모형과 분할/평가 조건 불일치: {key}")
    if old["input_hash"] != prep["input_hash"]:
        raise ValueError("기존 4모형과 입력 해시가 다릅니다")
    if not np.isclose(old["m3_beta"], current["m3_beta"], rtol=1e-10, atol=1e-12):
        raise ValueError("기존 M3와 beta가 다릅니다")
    for key, value in old["m3_audit"].items():
        if not np.isclose(value, current["m3_audit"][key], rtol=1e-9, atol=1e-11):
            raise ValueError(f"기존 M3와 그래프 진단 불일치: {key}")
    if not current["m4_weight_audit"]["invalid_extra_absent"]:
        raise ValueError("M4-A의 무효 학습행 raw=1 규칙 불일치")
    rows = reference["rows"]
    if len(rows) != 4 or {r["model_id"] for r in rows} != {M1, M3, M4_B, M5_B}:
        raise ValueError("기존 M1/M3/M4-B/M5-B 결과 누락 또는 중복")
    metric_keys = set(rows[0]["metrics"])
    if len(metric_keys) != 142:
        raise ValueError("기존 전체142지표가 없습니다")
    for row in rows:
        identity = row["identity"]
        if (row["seed"] != 49 or row["epochs"] != 300
                or identity["model_id"] != row["model_id"]
                or identity["input_hash"] != prep["input_hash"]
                or [h["epoch"] for h in row["training_history"]] != list(range(1, 301))
                or set(row["metrics"]) != metric_keys
                or not np.isfinite(list(row["metrics"].values())).all()):
            raise ValueError("기존 결과의 신원·학습완료·전체지표 검증 실패")
    return rows


def prepare(cfg=None, *, reference_path=REFERENCE_PATH):
    cfg = cfg or configure()
    if cfg != configure(out_dir=cfg.out_dir):
        raise ValueError("Dunnhumby seed49·300epoch 고정설정을 변경할 수 없습니다")
    reference = load_reference(reference_path)  # costly preparation must not precede this check
    prep = common.prepare(cfg)
    prep["reference_rows"] = validate_reference(reference, prep, cfg)
    prep["reference_protocol"] = reference["protocol"]
    prep["reference_path"] = str(reference_path)
    prep["reference_sha256"] = REFERENCE_SHA
    common.io._atomic_json(prep["run_dir"] / "reference_audit.json", dict(
        result_sha256=REFERENCE_SHA, path=str(reference_path),
        validation="input/config/split/population/M3/142metrics/300epoch history matched",
        retrain_references=False, reevaluate_references=False,
        reference_protocol=reference["protocol"]))
    print("[계획] 기존 M1·M3·M4-B·M5-B 재사용 / M4-A·M5-A만 각 300epoch 학습", flush=True)
    return prep


def _arm_path(prep, model_id):
    return prep["run_dir"] / "arms" / f"{model_id}_s49.json"


def load_completed_arm(prep, cfg, model_id):
    path = _arm_path(prep, model_id)
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    identity = row.get("identity", {})
    if (row.get("model_id") != model_id or row.get("seed") != 49
            or row.get("epochs") != cfg.epochs
            or identity.get("stage") != common.CODE_VERSION
            or identity.get("config_hash") != prep["config_hash"]
            or identity.get("source_revision") != prep["revision"]
            or identity.get("input_hash") != prep["input_hash"]
            or [h["epoch"] for h in row.get("training_history", [])] != list(range(1, 301))
            or len(row.get("metrics", {})) != 142
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise ValueError(f"완료된 {model_id} 결과 신원·전체지표 검증 실패")
    return row


def run_selected(cfg, prep, model_id):
    if model_id not in (M4_A, M5_A):
        raise ValueError("M4-A 또는 M5-A만 선택하세요")
    validate_reference(load_reference(prep["reference_path"]), prep, cfg)
    digest = hashlib.sha256(prep["m4_weights"].astype(np.float32).tobytes()).hexdigest()
    if digest != prep["protocol"]["m4_weight_audit"]["sha256"]:
        raise ValueError("prepare 이후 M4-A 가중치가 변경됐습니다")
    print(f"[추가학습] {model_id} / seed49 / 300epoch", flush=True)
    return common.run_arm(prep, cfg, model_id, 49)


def report(prep, cfg, rows):
    if (len(rows) != 6 or {r["model_id"] for r in rows} != set(MODELS)
            or any(r["seed"] != 49 or r["epochs"] != cfg.epochs for r in rows)):
        raise ValueError("여섯 모형의 고정점 결과가 모두 있어야 판독할 수 있습니다")
    metrics = list(rows[0]["metrics"])
    if len(metrics) != 142 or any(set(r["metrics"]) != set(metrics) for r in rows):
        raise ValueError("모형 간 전체 평가지표가 다릅니다")
    absolute = pd.DataFrame([dict(model_id=r["model_id"], seed=49, epoch=cfg.epochs,
                                 **r["metrics"]) for r in rows])
    index = absolute.set_index("model_id").loc[list(MODELS)]
    if not np.isfinite(index[metrics].to_numpy(float)).all():
        raise ValueError("평가지표 비유한값")
    pairs = ((M3,M1), (M4_A,M1), (M5_A,M1), (M5_A,M3), (M5_A,M4_A),
             (M4_B,M1), (M5_B,M1), (M5_B,M3), (M5_B,M4_B),
             (M4_A,M4_B), (M5_A,M5_B))
    comparisons = []
    for model, ref in pairs:
        for metric in metrics:
            value, base = float(index.at[model,metric]), float(index.at[ref,metric])
            comparisons.append(dict(model_id=model, reference=ref, seed=49,
                                    epoch=cfg.epochs, metric=metric, value=value,
                                    reference_value=base, delta=value-base,
                                    relative_change_pct=100*(value/base-1) if base else None))
    interactions = []
    for label, m4, m5 in (("A",M4_A,M5_A), ("B",M4_B,M5_B)):
        for metric in metrics:
            interactions.append(dict(
                variant=label, metric=metric, seed=49, epoch=cfg.epochs,
                m3_minus_m1=index.at[M3,metric]-index.at[M1,metric],
                m4_minus_m1=index.at[m4,metric]-index.at[M1,metric],
                m5_minus_m4=index.at[m5,metric]-index.at[m4,metric],
                interaction_absolute=(index.at[m5,metric]-index.at[M3,metric]
                                      -index.at[m4,metric]+index.at[M1,metric])))
    guards = {model: bool(all(index.at[model,m] >= .99*index.at[M1,m]
                             for m in common.ACCURACY))
              for model in (M3,M4_A,M5_A,M4_B,M5_B)}
    variants = {}
    for label, m4, m5 in (("A",M4_A,M5_A), ("B",M4_B,M5_B)):
        economic = all(index.at[m5,m] > max(index.at[x,m] for x in (M1,M3,m4))
                       for m in common.ECONOMIC)
        best_guard = all(index.at[m5,m] >= .99*max(index.at[M3,m],index.at[m4,m])
                         for m in common.ACCURACY)
        variants[label] = dict(
            m5_economic_above_m1_m3_own_m4=bool(economic),
            m5_accuracy_guard_vs_best_standalone=bool(best_guard),
            descriptive_combination_condition_met=bool(economic and guards[m5] and best_guard))
    reading = dict(
        complete=True, seed_count=1, evaluated_epoch=cfg.epochs,
        accuracy_guard_vs_m1=guards, variants=variants,
        a_b_selected=False, selection_requires_more_than_this_exposed_seed=True,
        single_seed_diagnostic_only=True, previously_exposed_test=True,
        significance_claim=False, clv_attribution_claim=False,
        causal_interaction_claim=False, final_ten_seed_report=False,
        no_post_test_selection=True)
    reused = {M1,M3,M4_B,M5_B}
    tables = dict(
        absolute=absolute, comparison=pd.DataFrame(comparisons),
        interaction=pd.DataFrame(interactions),
        diagnostics=pd.DataFrame([dict(model_id=r["model_id"], seed=49,
                                       reused=r["model_id"] in reused,
                                       **r["diagnostics"]) for r in rows]))
    root = prep["run_dir"] / "reports"
    paths = {key:str(root/f"{key}.csv") for key in tables}
    for key, table in tables.items():
        common.io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root/"result.json")
    common.io._atomic_json(Path(paths["json"]), dict(
        protocol=prep["protocol"], reference_protocol=prep["reference_protocol"],
        reference_sha256=prep["reference_sha256"], rows=rows,
        reading=reading, paths=paths))
    return dict(**tables, reading=reading, paths=paths)


def report_if_complete(cfg, prep):
    references = validate_reference(load_reference(prep["reference_path"]), prep, cfg)
    a_rows = [load_completed_arm(prep, cfg, model) for model in (M4_A,M5_A)]
    missing = [model for model, row in zip((M4_A,M5_A), a_rows) if row is None]
    if missing:
        return dict(complete=False, missing=missing,
                    message="다른 A arm 학습 완료 후 이 셀을 다시 실행하세요.")
    return report(prep, cfg, list(references)+a_rows)
