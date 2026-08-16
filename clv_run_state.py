"""Atomic progress, epoch checkpoints, and deterministic resume helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RunIdentity:
    stage: str
    model_id: str
    seed: int
    config_hash: str
    source_revision: str
    input_hash: str


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class ProgressStore:
    """One stage's resume state plus run-wide human-readable progress files."""

    _CSV_FIELDS = (
        "last_heartbeat_at",
        "stage",
        "model_id",
        "status",
        "epoch",
        "max_epoch",
        "batch",
        "batches",
        "best_epoch",
        "best_metric",
        "loss",
        "eta_sec",
        "checkpoint_path",
        "extra_json",
    )

    def __init__(
        self,
        root: str | Path,
        identity: RunIdentity,
        *,
        heartbeat_interval_sec: float = 60.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.heartbeat_interval_sec = float(heartbeat_interval_sec)
        safe = f"{identity.stage}_{identity.model_id}_s{identity.seed}"
        self.stage_path = self.root / "stages" / f"{safe}.json"
        self.latest_checkpoint = self.root / "resume" / f"{safe}_latest.pt"
        self.complete_path = self.root / "stages" / f"{safe}.completed.json"
        self.progress_json = self.root / "progress.json"
        self.progress_csv = self.root / "progress.csv"
        self._last_heartbeat_monotonic = -float("inf")

    def _validate_identity(self, actual: dict[str, Any]) -> None:
        expected = asdict(self.identity)
        if actual != expected:
            raise RuntimeError(
                "resume checkpoint identity mismatch: "
                f"expected={expected}, actual={actual}"
            )

    def _write_progress(self, status: str, fields: dict[str, Any]) -> dict[str, Any]:
        timestamp = _now()
        payload = {
            **asdict(self.identity),
            "status": status,
            "last_heartbeat_at": timestamp,
            "checkpoint_path": str(self.latest_checkpoint),
            **fields,
        }
        _atomic_json(self.progress_json, payload)
        _atomic_json(self.stage_path, payload)
        known = {name: payload.get(name, "") for name in self._CSV_FIELDS}
        extra = {key: value for key, value in payload.items() if key not in known}
        known["extra_json"] = json.dumps(extra, ensure_ascii=False, default=str)
        self.progress_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = self.progress_csv.exists()
        with self.progress_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(known)
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def mark_stage(self, status: str, **fields: Any) -> dict[str, Any]:
        return self._write_progress(status, fields)

    def heartbeat(self, **fields: Any) -> bool:
        now = time.monotonic()
        if now - self._last_heartbeat_monotonic < self.heartbeat_interval_sec:
            return False
        self._last_heartbeat_monotonic = now
        self._write_progress("running", fields)
        return True

    def save_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
        **epoch_state: Any,
    ) -> Path:
        payload = {
            "identity": asdict(self.identity),
            "model_state": clone_state(model),
            "optimizer_state": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            **epoch_state,
        }
        _atomic_torch(self.latest_checkpoint, payload)
        self._write_progress(
            "running",
            {
                "epoch": int(epoch_state["epoch"]),
                "best_epoch": int(epoch_state.get("best_epoch", 0)),
                "best_metric": float(epoch_state.get("best_metric", float("nan"))),
                "checkpoint_sha256": file_sha256(self.latest_checkpoint),
            },
        )
        return self.latest_checkpoint

    def restore_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        if not self.latest_checkpoint.exists():
            return None
        payload = torch.load(
            self.latest_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_identity(payload.get("identity", {}))
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_state = payload.get("cuda_rng_state", [])
        if torch.cuda.is_available() and cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
        excluded = {
            "identity",
            "model_state",
            "optimizer_state",
            "numpy_rng_state",
            "torch_rng_state",
            "cuda_rng_state",
        }
        state = {key: value for key, value in payload.items() if key not in excluded}
        state["next_epoch"] = int(payload["epoch"]) + 1
        return state

    def mark_complete(self, **fields: Any) -> dict[str, Any]:
        payload = self._write_progress("completed", fields)
        complete = {"identity": asdict(self.identity), **payload}
        _atomic_json(self.complete_path, complete)
        return payload

    def mark_failed(self, error: str, **fields: Any) -> dict[str, Any]:
        return self._write_progress("failed", {"error": str(error), **fields})

    def is_complete(self) -> bool:
        if not self.complete_path.exists():
            return False
        payload = json.loads(self.complete_path.read_text(encoding="utf-8"))
        self._validate_identity(payload.get("identity", {}))
        return payload.get("status") == "completed"

    def read_progress(self) -> dict[str, Any]:
        path = self.stage_path if self.stage_path.exists() else self.progress_json
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._validate_identity(
            {name: payload.get(name) for name in asdict(self.identity)}
        )
        return payload
