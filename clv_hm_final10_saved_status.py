"""Read H&M final10 progress and completed metrics without Torch or training."""

from __future__ import annotations

import json
import math
from pathlib import Path


VERSION = "clv-m5-user-centered-hm2y-final10-v1"
ROOT = "/content/drive/MyDrive/논문/data/results_v3_hm_m5c_lastweek_final10_v1"
MODELS = (
    "m1_bpr_k1_hm2y_lastweek_final10",
    "m3_centered_nv_bpr_k1_hm2y_lastweek_final10",
    "m5_m3nv_user_centered_m4_bpr_k1_hm2y_lastweek_final10",
)


def inspect_saved(root: str = ROOT) -> dict:
    rows, completed, protocols = [], [], []
    for path in sorted(Path(root).glob("*/protocol.json")):
        protocol = json.loads(path.read_text())
        if protocol.get("code_version") != VERSION:
            continue
        cfg = protocol.get("config", {})
        fixed = {"dataset": "hm", "epochs": 300, "batch_size": 131072,
                 "id_dim": 64, "n_layers": 2, "lr": .0005, "pref_reg": .001,
                 "negative_count": 1, "input_days": 365}
        if any(cfg.get(k) != v for k, v in fixed.items()):
            continue
        bounds = protocol.get("data_stats", {}).get("split_boundaries", {})
        if (str(bounds.get("train", {}).get("end_inclusive", ""))[:10] != "2020-09-15"
                or str(bounds.get("test", {}).get("end_inclusive", ""))[:10] != "2020-09-22"):
            continue
        run_dir = path.parent
        protocols.append(str(path))
        for seed in (49, 50):
            for model in MODELS:
                identity = {"stage": VERSION, "model_id": model, "seed": seed,
                            "config_hash": run_dir.name,
                            "source_revision": protocol["source_revision"],
                            "input_hash": protocol["input_hash"]}
                result = run_dir / "arms" / f"{model}_s{seed}.json"
                safe = f"{VERSION}_{model}_s{seed}"
                checkpoint = run_dir / "progress/resume" / f"{safe}_latest.pt"
                stage = run_dir / "progress/stages" / f"{safe}.json"
                row = {"run_hash": run_dir.name, "seed": seed, "model_id": model,
                       "status": "새 학습 필요(이 실행 폴더 범위)", "state_epoch": None,
                       "checkpoint_exists": checkpoint.is_file(), "checkpoint_epoch_verified": False}
                if result.is_file():
                    raw = json.loads(result.read_text())
                    if raw.get("identity") != identity or raw.get("epochs") != 300:
                        raise RuntimeError(f"완료 결과 신원 불일치: {result}")
                    metrics = raw.get("metrics", {})
                    if not metrics or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                              for v in metrics.values()):
                        raise RuntimeError(f"완료 결과 지표 누락/비유한: {result}")
                    row.update(status="완료 결과 재사용(추가 학습 없음)", state_epoch=300)
                    completed.append({"run_hash": run_dir.name, "seed": seed,
                                      "model_id": model, **metrics})
                elif checkpoint.is_file():
                    row["status"] = "재개 checkpoint 존재(실제 epoch는 학습러너가 확인)"
                    if stage.is_file():
                        state = json.loads(stage.read_text())
                        if any(state.get(k) != v for k, v in identity.items()):
                            raise RuntimeError(f"상태 JSON 신원 불일치: {stage}")
                        row["state_epoch"] = state.get("epoch")
                rows.append(row)
    if not protocols:
        raise FileNotFoundError(f"이 검색 범위에서 현재 H&M 최종 실행 protocol을 찾지 못했습니다: {root}")
    return {"status": rows, "completed_metrics": completed, "protocols": protocols,
            "checkpoint_loaded": False, "new_training": 0, "test_evaluations": 0,
            "final_decision_available": False,
            "note": "state_epoch는 작은 상태JSON의 진행epoch입니다. checkpoint 실물은 로드하지 않았습니다."}
