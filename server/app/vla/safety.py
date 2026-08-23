from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.robot.so101_spec import SO101_JOINTS


@dataclass
class SafetyResult:
    accepted: bool
    action: np.ndarray | None
    events: list[str]
    error: str | None = None


class SafetyFilter:
    def __init__(
        self,
        *,
        max_joint_delta: float = 0.12,
        max_velocity: float = 3.6,
        clamp_tolerance: float = 0.025,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.max_joint_delta = max_joint_delta
        self.max_velocity = max_velocity
        self.clamp_tolerance = clamp_tolerance
        self.timeout_seconds = timeout_seconds
        self.emergency_stopped = False

    def emergency_stop(self) -> None:
        self.emergency_stopped = True

    def reset(self) -> None:
        self.emergency_stopped = False

    def filter(
        self,
        action: np.ndarray | list[float],
        current: np.ndarray | list[float],
        *,
        dt: float = 1 / 30,
        age_seconds: float = 0.0,
    ) -> SafetyResult:
        if self.emergency_stopped:
            return SafetyResult(False, None, ["ACTION_REJECTED"], "EMERGENCY_STOP")
        if age_seconds > self.timeout_seconds:
            return SafetyResult(False, None, ["ACTION_REJECTED"], "ACTION_TIMEOUT")
        target = np.asarray(action, dtype=np.float32)
        state = np.asarray(current, dtype=np.float32)
        if target.shape != (6,) or state.shape != (6,):
            return SafetyResult(False, None, ["ACTION_REJECTED"], "ACTION_SHAPE_MISMATCH")
        if not np.all(np.isfinite(target)):
            return SafetyResult(False, None, ["ACTION_REJECTED"], "ACTION_NON_FINITE")
        lower = np.asarray(SO101_JOINTS.lower, dtype=np.float32)
        upper = np.asarray(SO101_JOINTS.upper, dtype=np.float32)
        excess = np.maximum(lower - target, target - upper)
        if np.any(excess > self.clamp_tolerance):
            return SafetyResult(False, None, ["ACTION_REJECTED"], "ACTION_JOINT_LIMIT")
        events: list[str] = []
        safe = np.clip(target, lower, upper)
        if not np.array_equal(safe, target):
            events.append("ACTION_CLAMPED")
        allowed_delta = min(self.max_joint_delta, self.max_velocity * dt)
        delta = safe - state
        # SmolVLA emits absolute joint-position targets. A valid target may be
        # far from the current calibrated pose on the first chunk, so rate-limit
        # it into a safe trajectory instead of rejecting the whole policy run.
        if np.any(np.abs(delta) > allowed_delta * 4):
            events.append("ACTION_TARGET_RAMPED")
        clipped_delta = np.clip(delta, -allowed_delta, allowed_delta)
        if not np.array_equal(clipped_delta, delta):
            events.append("ACTION_CLAMPED")
        return SafetyResult(True, state + clipped_delta, list(dict.fromkeys(events)))
