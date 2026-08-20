from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    error: str | None = None


class SafetyValidator:
    def __init__(self, max_steps: int = 15) -> None:
        self.max_steps = max_steps

    def validate_move(self, x: int, y: int, high: bool, step: int, stopped: bool = False) -> SafetyResult:
        if stopped:
            return SafetyResult(False, "SESSION_STOPPED")
        if step >= self.max_steps:
            return SafetyResult(False, "MAX_AGENT_STEPS")
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, int) or not isinstance(y, int):
            return SafetyResult(False, "COORDINATES_MUST_BE_INTEGERS")
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            return SafetyResult(False, "COORDINATES_OUT_OF_BOUNDS")
        if not isinstance(high, bool):
            return SafetyResult(False, "HIGH_MUST_BE_BOOLEAN")
        return SafetyResult(True)

    def validate_gripper(self, opened: bool, step: int, stopped: bool = False) -> SafetyResult:
        if stopped:
            return SafetyResult(False, "SESSION_STOPPED")
        if step >= self.max_steps:
            return SafetyResult(False, "MAX_AGENT_STEPS")
        if not isinstance(opened, bool):
            return SafetyResult(False, "OPENED_MUST_BE_BOOLEAN")
        return SafetyResult(True)

    def validate_semantic_skill(
        self, value: object, allowed: set[str], step: int, stopped: bool = False
    ) -> SafetyResult:
        if stopped:
            return SafetyResult(False, "SESSION_STOPPED")
        if step >= self.max_steps:
            return SafetyResult(False, "MAX_AGENT_STEPS")
        if not isinstance(value, str) or value not in allowed:
            return SafetyResult(False, "UNKNOWN_SCENE_OBJECT")
        return SafetyResult(True)
