from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.robot.backends.so101_mujoco import SO101MujocoBackend


@dataclass
class ExpertEpisode:
    success: bool
    actions: list[np.ndarray]
    frames: list[dict[str, Any]]
    error: str | None = None


class SO101ExpertPolicy:
    """Ground-truth classical baseline and demonstration generator only.

    This policy is intentionally separate from the VLA observation path. It may
    use simulator poses and IK, but its exported actions are ordinary 6D SO-101
    joint-position commands executed continuously at the policy frequency.
    """

    def __init__(self, backend: SO101MujocoBackend) -> None:
        self.backend = backend

    def run(self, object_id: str = "red_cube", container_id: str = "blue_box") -> ExpertEpisode:
        frames: list[dict[str, Any]] = []
        pick = self.backend.pick_object(object_id)
        frames.extend(pick.frames or [])
        if not pick.success:
            return ExpertEpisode(False, self._actions(frames), frames, pick.error)
        place = self.backend.place_object(container_id)
        frames.extend(place.frames or [])
        if not place.success:
            return ExpertEpisode(False, self._actions(frames), frames, place.error)
        verified = self.backend.verify_task()
        return ExpertEpisode(
            verified,
            self._actions(frames),
            frames,
            None if verified else "TASK_VERIFICATION_FAILED",
        )

    @staticmethod
    def _actions(frames: list[dict[str, Any]]) -> list[np.ndarray]:
        return [
            np.asarray(frame["robot"]["joints"], dtype=np.float32)
            for frame in frames
        ]
