from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SO101JointSpec:
    names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    lower: tuple[float, ...] = (-1.919862, -1.745329, -1.69, -1.658063, -2.743847, -0.174533)
    upper: tuple[float, ...] = (1.919862, 1.745329, 1.69, 1.658063, 2.841206, 1.745329)
    home: tuple[float, ...] = (0.0, -0.75, 1.20, 0.85, 0.0, 1.45)

    @property
    def size(self) -> int:
        return len(self.names)

    def lower_array(self) -> np.ndarray:
        return np.asarray(self.lower, dtype=np.float32)

    def upper_array(self) -> np.ndarray:
        return np.asarray(self.upper, dtype=np.float32)

    def home_array(self) -> np.ndarray:
        return np.asarray(self.home, dtype=np.float64)


SO101_JOINTS = SO101JointSpec()

