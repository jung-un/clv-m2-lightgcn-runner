"""Train-only, no-retraining diagnostics for the Dunnhumby M3 CLV-NV graph."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3
from clv_m3_nv_diagnostics import analyze_clv_nv_graph
from lightgcn_clv_m3_nv import (
    _activate,
    configure_m3_clv_nv_dunnhumby_run,
    validate_screening_config,
)


CODE_VERSION = "m3-clv-nv-graph-diagnostic-v1"


def _prepare_train_graph(cfg: dict) -> tuple[pd.DataFrame, object, int, int]:
    """Prepare train-only graph inputs; never call a model training entry point."""
    _activate(cfg)
    prepared = v3.prepare_data(v3.CFG, v3.DCFG)
    graph = prepared.get("clv_nv_graph")
    if graph is None:
        raise RuntimeError("GRAPH_MODE=clv_nv did not produce CLV-NV graph weights")
    return (
        prepared["train"],
        graph,
        int(prepared["n_users"]),
        int(prepared["n_items"]),
    )


def _diagnostic_out_dir(cfg: dict) -> Path:
    return Path(cfg["OUT_DIR"]) / "graph_diagnostics"


def _persist_report(report: dict, cfg: dict) -> dict[str, str]:
    out_dir = _diagnostic_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "m3_clv_nv_graph_diagnostic"
    paths = {
        "summary_json": out_dir / f"{stem}_summary.json",
        "correlations_csv": out_dir / f"{stem}_correlations.csv",
        "weight_deciles_csv": out_dir / f"{stem}_weight_deciles.csv",
        "top_items_csv": out_dir / f"{stem}_top_items.csv",
    }
    payload = {
        "code_version": CODE_VERSION,
        "scope": "train graph only; no model training; no validation/test labels",
        "dataset": cfg["DATASET"],
        "graph_mode": cfg["GRAPH_MODE"],
        "summary": report["summary"],
    }
    paths["summary_json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["correlations"].to_csv(paths["correlations_csv"], index=False)
    report["weight_deciles"].to_csv(paths["weight_deciles_csv"], index=False)
    report["top_items"].to_csv(paths["top_items_csv"], index=False)
    return {name: str(path) for name, path in paths.items()}


def run_graph_diagnostics(cfg: dict | None = None) -> dict:
    """Diagnose the current M3 graph without fitting or scoring any model."""
    cfg = configure_m3_clv_nv_dunnhumby_run() if cfg is None else dict(cfg)
    validate_screening_config(cfg)
    train, graph, n_users, n_items = _prepare_train_graph(cfg)
    report = analyze_clv_nv_graph(
        train,
        graph,
        n_users=n_users,
        n_items=n_items,
    )
    report["paths"] = _persist_report(report, cfg)
    print("M3 CLV-NV train-graph diagnostic summary:")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\nSpearman correlations:")
    print(report["correlations"].to_string(index=False))
    print("\nWeight deciles:")
    print(report["weight_deciles"].to_string(index=False))
    print("\nTop amplified items:")
    print(report["top_items"].to_string(index=False))
    print("\nSaved files:", report["paths"])
    return report


if __name__ == "__main__":
    print(
        "This entry point only prints its purpose. "
        "Call run_graph_diagnostics() explicitly to read train data."
    )
