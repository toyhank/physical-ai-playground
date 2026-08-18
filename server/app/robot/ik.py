from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IKResult:
    success: bool
    joints: np.ndarray
    iterations: int
    error: float


class PlanarDampedLeastSquaresIK:
    """Stable position-only DLS IK used by the MVP Franka abstraction."""

    def __init__(
        self,
        link_lengths: tuple[float, float, float] = (0.33, 0.28, 0.18),
        damping: float = 0.06,
        tolerance: float = 0.004,
        max_iterations: int = 160,
        max_dq: float = 0.12,
    ) -> None:
        self.links = np.asarray(link_lengths, dtype=float)
        self.damping = damping
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.max_dq = max_dq
        self.lower = np.asarray((-2.9, -2.1, -2.9), dtype=float)
        self.upper = np.asarray((2.9, 2.1, 2.9), dtype=float)

    def forward(self, joints: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(joints)
        return np.asarray(
            [np.sum(self.links * np.cos(cumulative)), np.sum(self.links * np.sin(cumulative))]
        )

    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(joints)
        jacobian = np.zeros((2, 3), dtype=float)
        for column in range(3):
            jacobian[0, column] = -np.sum(self.links[column:] * np.sin(cumulative[column:]))
            jacobian[1, column] = np.sum(self.links[column:] * np.cos(cumulative[column:]))
        return jacobian

    def solve(self, target: np.ndarray, initial: np.ndarray | None = None) -> IKResult:
        target = np.asarray(target, dtype=float)
        joints = np.asarray(initial if initial is not None else (0.3, 0.8, -1.0), dtype=float).copy()
        for iteration in range(1, self.max_iterations + 1):
            error_vector = target - self.forward(joints)
            error = float(np.linalg.norm(error_vector))
            if error <= self.tolerance:
                return IKResult(True, joints, iteration, error)
            jacobian = self.jacobian(joints)
            damped = jacobian @ jacobian.T + (self.damping**2) * np.eye(2)
            dq = jacobian.T @ np.linalg.solve(damped, error_vector)
            dq = np.clip(dq, -self.max_dq, self.max_dq)
            joints = np.clip(joints + dq, self.lower, self.upper)
        error = float(np.linalg.norm(target - self.forward(joints)))
        return IKResult(False, joints, self.max_iterations, error)

