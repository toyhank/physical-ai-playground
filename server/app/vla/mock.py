from __future__ import annotations

from typing import Any

import numpy as np


class MockVLAProvider:
    """Deterministic observation-only VLA substitute for integration tests.

    It deliberately reads only the public VLA feature keys. It never receives
    scene state, MuJoCo poses, object IDs, segmentation, or IK targets.
    """

    name = "mock-vla"

    def __init__(self, chunk_size: int = 8) -> None:
        self.chunk_size = chunk_size
        self._phase = 0

    def infer(self, observation: dict[str, Any], task: str) -> np.ndarray:
        required = {
            "observation.images.scene",
            "observation.images.wrist",
            "observation.state",
        }
        if set(observation) != required:
            raise ValueError("VLA_OBSERVATION_CONTAINS_FORBIDDEN_FEATURES")
        if not task.strip():
            raise ValueError("VLA_TASK_EMPTY")
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (6,):
            raise ValueError("VLA_STATE_SHAPE_MISMATCH")
        # A smooth, bounded action chunk is enough to exercise the complete
        # observation -> policy -> safety -> queue -> physics integration path.
        direction = -1.0 if self._phase % 2 else 1.0
        offsets = np.linspace(0.002, 0.016, self.chunk_size, dtype=np.float32)
        chunk = np.repeat(state[None, :], self.chunk_size, axis=0)
        chunk[:, 0] += direction * offsets
        chunk[:, 4] -= direction * offsets * 0.5
        self._phase += 1
        return chunk

    def reset_session(self) -> None:
        self._phase = 0
