from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_id: str, prompt: str, model: str, root: Path | None = None) -> None:
        data_root = root or Path(__file__).resolve().parents[3] / "data" / "runs"
        self.directory = data_root / run_id
        self.observations = self.directory / "observations"
        self.observations.mkdir(parents=True, exist_ok=True)
        self.events_file = self.directory / "events.jsonl"
        self.metadata = {
            "run_id": run_id,
            "prompt": prompt,
            "model": model,
            "started_at": datetime.now(UTC).isoformat(),
            "duration_seconds": None,
            "steps": 0,
            "task_success": False,
            "failure_reason": None,
        }
        self._started = datetime.now(UTC)
        self._write_metadata()

    def event(self, payload: dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), **payload}
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if payload.get("type") == "tool":
            self.metadata["steps"] += 1
            self._write_metadata()

    def observation(self, index: int, png_bytes: bytes, camera: str = "scene") -> None:
        (self.observations / f"{camera}_{index:03d}.png").write_bytes(png_bytes)

    def finish(self, success: bool, failure_reason: str | None = None) -> None:
        self.metadata["duration_seconds"] = round((datetime.now(UTC) - self._started).total_seconds(), 3)
        self.metadata["task_success"] = success
        self.metadata["failure_reason"] = failure_reason
        self.metadata["finished_at"] = datetime.now(UTC).isoformat()
        self._write_metadata()

    def _write_metadata(self) -> None:
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
