from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import numpy as np


class SmolVLAClient:
    name = "smolvla"

    def __init__(self, host: str, timeout_seconds: float = 30.0) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session_id = uuid4().hex

    def infer(self, observation: dict[str, Any], task: str) -> np.ndarray:
        response = httpx.post(
            f"{self.host}/infer",
            json={
                "session_id": self.session_id,
                "task": task,
                "scene_png_base64": observation["observation.images.scene"],
                "wrist_png_base64": observation["observation.images.wrist"],
                "state": observation["observation.state"],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        array = np.asarray(response.json()["action_chunk"], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 6:
            raise ValueError("SMOLVLA_ACTION_CHUNK_SHAPE_MISMATCH")
        return array

    def reset_session(self) -> None:
        response = httpx.post(
            f"{self.host}/reset",
            json={"session_id": self.session_id},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
