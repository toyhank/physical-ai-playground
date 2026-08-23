from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.robot.backends import SO101MujocoBackend
from app.robot.expert import SO101ExpertPolicy


TRAINING_FEATURE_KEYS = frozenset(
    {
        "observation.images.scene",
        "observation.images.wrist",
        "observation.state",
        "action",
        "task",
    }
)


def build_training_frame(
    observation: dict[str, Any], action: np.ndarray, task: str
) -> dict[str, Any]:
    """Whitelist policy features so simulator ground truth cannot leak."""
    frame = {
        "observation.images.scene": observation["observation.images.scene"],
        "observation.images.wrist": observation["observation.images.wrist"],
        "observation.state": np.asarray(observation["observation.state"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if set(frame) != TRAINING_FEATURE_KEYS:
        raise RuntimeError("TRAINING_FEATURE_WHITELIST_BROKEN")
    return frame


def _decode_png(value: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))


@dataclass
class CollectedEpisode:
    success: bool
    frames: list[dict[str, Any]]
    error: str | None


class SO101DemonstrationCollector:
    def collect(
        self,
        backend: SO101MujocoBackend,
        *,
        object_id: str = "red_cube",
        task: str = "Pick up the red cube and place it in the blue box.",
    ) -> CollectedEpisode:
        frames: list[dict[str, Any]] = []

        def observe(action: np.ndarray) -> None:
            observation = backend.get_observation()
            observation["observation.images.scene"] = _decode_png(
                observation["observation.images.scene"]
            )
            observation["observation.images.wrist"] = _decode_png(
                observation["observation.images.wrist"]
            )
            frames.append(build_training_frame(observation, action, task))

        backend.frame_observer = observe
        try:
            result = SO101ExpertPolicy(backend).run(object_id)
        finally:
            backend.frame_observer = None
        return CollectedEpisode(result.success, frames if result.success else [], result.error)


def lerobot_v3_features() -> dict[str, Any]:
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    return {
        "observation.images.scene": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (6,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (6,), "names": joint_names},
    }
