"""Read completed development JSONs only. Never imports a training runner/torch.

No Drive writes, checkpoint deserialization, training, or missing-arm execution.
Cross-run provenance is retained; averages and success decisions are not inferred.
"""
from pathlib import Path
import json
import math
import pandas as pd


PATTERNS = (
    "results_v3_dunnhumby_m4_k1_assignment_control_development_multiseed_v1",
    "results_v3_dunnhumby_clv_component_recheck_dev*_v1",
    "results_v3_dunnhumby_history_m5_strength_*_v1",
    "results_v3_hm_m4_k1_assignment_control_hm2y*",
    "results_v3_hm_clv_m2_training_budget_seed*_v1",
)
SEEDS = (42, 43, 44)


def metric_values(row):
    return {k: float(v) for k, v in row.items()
            if ("@" in k or k == "user_value_tendency_recommended_price_alignment")
            and isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v)}


def model_role(name):
    if name.startswith("m1"):
        return "M1"
    if name.startswith("m4"):
        return "M4 순열" if "shuffle" in name else "M4 실제"
    if name.startswith("m5"):
        return "M5"
    if name.startswith("m2"):
        return "M2"
    return "기타"


def readout(data_root):
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Drive data 폴더를 확인하세요: {root}")
    folders = sorted({p for pattern in PATTERNS for p in root.glob(pattern) if p.is_dir()})
    records, warnings, checkpoints = {}, [], []

    def read(path):
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("64 MB 초과 JSON은 자동 로드하지 않음")
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError) as exc:
            warnings.append({"source": str(path), "warning": str(exc)})
            return {}

    def add(folder, run_hash, row, path, parent=None):
        parent = parent or {}
        identity = row.get("identity", {})
        cfg = identity.get("protocol", parent.get("config", {}))
        name = row.get("model_id", "")
        seed = row.get("seed", cfg.get("seed"))
        metrics = metric_values(row.get("metrics", row))
        if seed not in SEEDS or not name or not metrics:
            return
        role = model_role(name)
        if role == "기타":
            return
        mode = (cfg.get("m4_mode") or
                ("complementary" if "component_recheck" in folder.name else "original"))
        if "training_budget" in folder.name:
            mode = "M4 없음"
        key = (str(folder), run_hash, name, seed)
        item = records.setdefault(key, {
            "dataset": "hm" if folder.name.startswith("results_v3_hm_") else "dunnhumby",
            "run": str(folder / run_hash), "seed": seed, "model_id": name,
            "role": role, "m4_mode": mode, "config": {}, "sources": [],
            "metrics": {}, "diagnostics": {}, "checkpoint": "",
            "split": "기록 없음", "final_epoch": None,
        })
        for metric, value in metrics.items():
            if metric in item["metrics"] and item["metrics"][metric] != value:
                item["conflict"] = True
                warnings.append({"source": str(path), "warning": f"중복 결과 충돌: {name} s{seed} {metric}"})
        item["metrics"].update(metrics)
        item["config"].update(cfg)
        item["sources"].append(str(path))
        item["diagnostics"].update(row.get("diagnostics", {}))
        item["checkpoint"] = row.get("checkpoint") or item["checkpoint"]
        item["split"] = identity.get("split") or row.get("split") or parent.get("preflight", {}).get("split") or item["split"]
        item["final_epoch"] = row.get("final_epoch", cfg.get("epochs", item["final_epoch"]))

    for folder in folders:
        # Consolidated reports supply config; per-arm files also work when a run is unfinished.
        for path in sorted(folder.glob("*.json")):
            payload = read(path)
            run_hash = path.stem.rsplit("_", 1)[-1]
            for row in payload.get("absolute_rows", []):
                if isinstance(row, dict):
                    add(folder, run_hash, row, path, payload)
        for path in sorted(folder.glob("arms/*/*.json")):
            payload = read(path)
            add(folder, path.parent.name, payload, path)
        # Metadata only: never torch.load. A checkpoint may be unfinished or incompatible.
        for path in folder.rglob("*.pt"):
            try:
                checkpoints.append({"dataset": "hm" if folder.name.startswith("results_v3_hm_") else "dunnhumby",
                                    "path": str(path), "size_MB": path.stat().st_size / 1024**2,
                                    "status": "파일 존재만 확인; 내용/epoch/호환성 미검증"})
            except OSError as exc:
                warnings.append({"source": str(path), "warning": str(exc)})

    inventory, absolute, comparisons = [], [], []
    for item in records.values():
        meta = {k: item[k] for k in ("dataset", "run", "seed", "model_id", "role", "m4_mode", "split", "final_epoch")}
        inventory.append({**meta, "metadata_status": "설정 있음; 입력/체크포인트 호환성 미검증" if item["config"] else "설정 메타데이터 없음",
                          "conflict": item.get("conflict", False),
                          "config": json.dumps(item["config"], ensure_ascii=False, sort_keys=True),
                          "checkpoint": item["checkpoint"], "sources": " | ".join(item["sources"]),
                          "diagnostics": json.dumps(item["diagnostics"], ensure_ascii=False)})
        absolute.append({**meta, **item["metrics"]})
        if item.get("conflict"):
            continue
        targets = ("M1", "M4 실제", "M2") if item["role"] == "M5" else ("M1",)
        for target in targets:
            if item["role"] == target:
                continue
            candidates = [r for r in records.values() if r["run"] == item["run"]
                          and r["seed"] == item["seed"] and r["role"] == target
                          and not r.get("conflict")]
            # Multiple rho M2 arms must not be picked arbitrarily.
            if len(candidates) != 1:
                continue
            base = candidates[0]
            for metric in sorted(base["metrics"].keys() & item["metrics"].keys()):
                b, v = base["metrics"][metric], item["metrics"][metric]
                comparisons.append({**meta, "reference": base["model_id"], "metric": metric,
                                    "reference_value": b, "value": v, "delta": v-b,
                                    "relative_change_pct": 100*(v-b)/b if b else float("nan")})
    presence = []
    for seed in SEEDS:
        for role in ("M1", "M4 실제", "M4 순열"):
            found = [r for r in records.values() if r["dataset"] == "hm" and r["seed"] == seed and r["role"] == role]
            presence.append({"seed": seed, "role": role, "completed_result_count": len(found),
                             "status": "결과 발견" if found else "검색 범위 내 완료 결과 미발견 — 추가 학습 안 함"})
    return {"inventory": pd.DataFrame(inventory), "absolute": pd.DataFrame(absolute),
            "within_run_comparison": pd.DataFrame(comparisons), "hm_presence": pd.DataFrame(presence),
            "checkpoints": pd.DataFrame(checkpoints), "warnings": pd.DataFrame(warnings)}
