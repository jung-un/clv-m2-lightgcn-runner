"""No-retraining decomposition of the conditioned category/price-history M2.

The diagnostic reuses one trained seed-42 historical-development checkpoint and
re-scores six deterministic views.  It never updates parameters, selects an
epoch, or constructs the protected final test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_conditioned_category_price_history as runner
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-conditioned-category-price-history-decomposition-v1"
VIEW_MODES = (
    "id_only",
    "id_category_unit",
    "id_price_unit",
    "learned_full",
    "equal_mix",
    "shuffled_condition",
)


@dataclass(frozen=True)
class DecompositionDiagnosticConfig:
    out_dir: str = ""
    baseline_result_dir: str = ""
    m2_checkpoint: str = ""
    shuffle_seed: int = 20260830


def configure_decomposition_diagnostic(**overrides) -> DecompositionDiagnosticConfig:
    defaults = runner.configure_conditioned_history_run()
    values = {
        "out_dir": defaults.out_dir,
        "baseline_result_dir": defaults.baseline_result_dir,
    }
    values.update(overrides)
    cfg = DecompositionDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: DecompositionDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "checkpoint_selection": False,
        "split": "historical_development_days_684_690",
        "final_test_constructed": False,
        "holdout_constructed": False,
        "views": list(VIEW_MODES),
        "view_definition": {
            "id_only": "jointly trained 64-dimensional ID block only",
            "id_category_unit": "ID plus category-history branch fully on; price off",
            "id_price_unit": "ID plus price-history branch fully on; category off",
            "learned_full": "checkpoint's learned user-specific two-way mixture",
            "equal_mix": "same checkpoint with category=0.5 and price=0.5",
            "shuffled_condition": "learned mixture reassigned across users",
        },
        "shuffle_seed": cfg.shuffle_seed,
        "interpretation": (
            "descriptive checkpoint decomposition only; no parameter is updated and "
            "no view is a new trained model"
        ),
        "statistical_note": "seed 42 descriptive diagnostic; no significance claim",
        "out_dir": cfg.out_dir,
    }


class _FixedEmbeddingView:
    def __init__(self, user: torch.Tensor, item: torch.Tensor):
        self.user = user
        self.item = item

    def embeddings(self, need_value: bool = True):
        user_zero = self.user.new_zeros((self.user.shape[0], 1))
        item_zero = self.item.new_zeros((self.item.shape[0], 1))
        return self.user, self.item, user_zero, item_zero


def _checkpoint_record(root: Path) -> tuple[Path, dict]:
    pattern = f"{runner.MODEL_ID}_s42.json"
    candidates = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for result_path in candidates:
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = Path(record["checkpoint"])
            if not checkpoint.exists():
                checkpoint = result_path.with_suffix(".pt")
            if not checkpoint.exists():
                continue
            expected_sha = record.get("checkpoint_sha256")
            if expected_sha and file_sha256(checkpoint) != expected_sha:
                continue
            return checkpoint, record
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    raise FileNotFoundError(f"{root} 아래에서 seed-42 M2 checkpoint를 찾지 못했습니다")


def _load_model(prepared: dict, runner_cfg, cfg: DecompositionDiagnosticConfig):
    if cfg.m2_checkpoint:
        checkpoint = Path(cfg.m2_checkpoint)
        record = {}
    else:
        checkpoint, record = _checkpoint_record(Path(cfg.out_dir))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    if payload.get("model_id") != runner.MODEL_ID:
        raise RuntimeError(f"다른 model checkpoint입니다: {payload.get('model_id')!r}")
    if payload.get("input_hash") != prepared["input_hash"]:
        raise RuntimeError("checkpoint와 현재 입력 데이터 해시가 다릅니다")
    model, _ = runner._build_model(prepared, runner_cfg)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, checkpoint, record


@torch.no_grad()
def _propagate_with_gate(
    model,
    gate: torch.Tensor,
    *,
    category_on: bool = True,
    price_on: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate.shape != (model.n_users, 2):
        raise ValueError(f"gate shape 오류: {tuple(gate.shape)}")
    category = model._unit_rows(model.category_embedding.weight)
    history_category = torch.sparse.mm(model.category_history, category)
    history_price = torch.sparse.mm(model.price_history, category)
    valid = model.auxiliary_valid[:, None]
    history_category = history_category * valid
    history_price = history_price * valid
    category_user = gate[:, :1] * history_category if category_on else torch.zeros_like(history_category)
    price_user = gate[:, 1:] * history_price if price_on else torch.zeros_like(history_price)
    user_aux = torch.cat([category_user, price_user], dim=1)
    item_category = category[model.item_category]
    item_price = model.item_price_signal[:, None] * item_category
    if not category_on:
        item_category = torch.zeros_like(item_category)
    if not price_on:
        item_price = torch.zeros_like(item_price)
    item_aux = torch.cat([item_category, item_price], dim=1)
    scale = float(np.sqrt(model.rho))
    current = torch.cat(
        [
            torch.cat([model.E_u.weight, scale * user_aux], dim=1),
            torch.cat([model.E_i.weight, scale * item_aux], dim=1),
        ],
        dim=0,
    )
    total = current
    for _ in range(model.n_layers):
        current = torch.sparse.mm(model.adj, current)
        total = total + current
    total = total / (model.n_layers + 1)
    return total[: model.n_users], total[model.n_users :]


@torch.no_grad()
def decomposition_views(model, shuffle_seed: int) -> tuple[dict, dict]:
    learned = model._gate()
    equal = torch.full_like(learned, 0.5)
    category = torch.zeros_like(learned)
    category[:, 0] = 1.0
    price = torch.zeros_like(learned)
    price[:, 1] = 1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed))
    permutation = torch.randperm(model.n_users, generator=generator).to(learned.device)
    shuffled = learned.index_select(0, permutation)

    id_user, id_item = model.id_only_embeddings()
    views = {
        "id_only": (id_user, id_item),
        "id_category_unit": _propagate_with_gate(
            model, category, category_on=True, price_on=False
        ),
        "id_price_unit": _propagate_with_gate(
            model, price, category_on=False, price_on=True
        ),
        "learned_full": _propagate_with_gate(model, learned),
        "equal_mix": _propagate_with_gate(model, equal),
        "shuffled_condition": _propagate_with_gate(model, shuffled),
    }
    gates = {
        "id_category_unit": category,
        "id_price_unit": price,
        "learned_full": learned,
        "equal_mix": equal,
        "shuffled_condition": shuffled,
    }
    return views, gates


def _view_metrics(views: dict, prepared: dict) -> pd.DataFrame:
    rows = []
    for name in VIEW_MODES:
        metrics, _ = moe._flat_evaluation(
            _FixedEmbeddingView(*views[name]),
            0.0,
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"],
            per_user=False,
        )
        rows.append({"view": name, **test10._public_metrics(metrics)})
    return pd.DataFrame(rows)


def _numeric_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index("view")
    rows = []
    for reference in ("id_only", "learned_full"):
        for view in VIEW_MODES:
            if view == reference:
                continue
            for metric in indexed.columns:
                reference_value = indexed.at[reference, metric]
                model_value = indexed.at[view, metric]
                if not isinstance(reference_value, (int, float, np.number)):
                    continue
                if not isinstance(model_value, (int, float, np.number)):
                    continue
                delta = float(model_value) - float(reference_value)
                rows.append(
                    {
                        "view": view,
                        "reference": reference,
                        "metric": metric,
                        "reference_value": float(reference_value),
                        "model_value": float(model_value),
                        "absolute_delta": delta,
                        "relative_change_pct": (
                            100.0 * delta / float(reference_value)
                            if float(reference_value) != 0.0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _gate_summary(gates: dict) -> pd.DataFrame:
    rows = []
    for name, gate in gates.items():
        values = gate.detach().cpu().numpy()
        rows.append(
            {
                "view": name,
                "category_mean": float(values[:, 0].mean()),
                "category_std": float(values[:, 0].std()),
                "category_min": float(values[:, 0].min()),
                "category_max": float(values[:, 0].max()),
                "price_mean": float(values[:, 1].mean()),
                "price_std": float(values[:, 1].std()),
                "price_min": float(values[:, 1].min()),
                "price_max": float(values[:, 1].max()),
            }
        )
    return pd.DataFrame(rows)


def _core_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    core = {
        "recall@10",
        "ndcg@10",
        "price_purchase_amount_weighted_hit@10",
        "recall@20",
        "ndcg@20",
        "price_purchase_amount_weighted_hit@20",
        "recall@50",
        "ndcg@50",
        "price_purchase_amount_weighted_hit@50",
        "coverage@10",
        "n_distinct@10",
        "top10_share@10",
    }
    return comparisons[
        (comparisons["reference"] == "id_only") & comparisons["metric"].isin(core)
    ].reset_index(drop=True)


def run_conditioned_history_decomposition(
    cfg: DecompositionDiagnosticConfig | None = None,
) -> dict:
    cfg = cfg or configure_decomposition_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = runner.configure_conditioned_history_run(
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )
    prepared = runner._prepare(runner_cfg)
    model, checkpoint, record = _load_model(prepared, runner_cfg, cfg)
    views, gates = decomposition_views(model, cfg.shuffle_seed)

    full_user, full_item, _, _ = model.embeddings()
    torch.testing.assert_close(views["learned_full"][0], full_user)
    torch.testing.assert_close(views["learned_full"][1], full_item)

    view_metrics = _view_metrics(views, prepared)
    comparisons = _numeric_comparisons(view_metrics)
    gate_summary = _gate_summary(gates)
    core = _core_table(comparisons)

    root = Path(cfg.out_dir) / "checkpoint_diagnostics" / "category_price_decomposition"
    paths = {
        "view_metrics_csv": root / "m2_conditioned_history_view_metrics.csv",
        "comparison_csv": root / "m2_conditioned_history_view_comparisons.csv",
        "gate_summary_csv": root / "m2_conditioned_history_gate_summary.csv",
        "json": root / "m2_conditioned_history_decomposition.json",
    }
    test10._atomic_csv(paths["view_metrics_csv"], view_metrics)
    test10._atomic_csv(paths["comparison_csv"], comparisons)
    test10._atomic_csv(paths["gate_summary_csv"], gate_summary)
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing seed-42 checkpoint only; no training or selection",
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "result_record_model_id": record.get("model_id"),
        },
        "view_metrics": view_metrics.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "gate_summary": gate_summary.to_dict("records"),
        "interpretation_rules": [
            "id_category_unit above id_price_unit means the category-history relation is stronger under full activation",
            "equal_mix above learned_full means learned mixture collapse is harmful",
            "shuffled_condition matching learned_full means user assignment is not materially used",
            "all conclusions are descriptive and do not claim significance",
        ],
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)

    result = {
        "view_metrics": view_metrics,
        "comparisons": comparisons,
        "gate_summary": gate_summary,
        "core": core,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    print("\n1) ID / 상품군 / 가격 / 결합 방식별 전체·CLV 구간 성과")
    print(view_metrics.to_string(index=False))
    print("\n2) ID-only 대비 핵심 변화")
    print(core.to_string(index=False))
    print("\n3) 실제 적용된 결합 비율")
    print(gate_summary.to_string(index=False))
    print("\n저장 파일")
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    print("No training is started automatically. Call run_conditioned_history_decomposition().")
