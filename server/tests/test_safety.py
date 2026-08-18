from __future__ import annotations

from app.robot.safety import SafetyValidator


def test_safety_accepts_valid_move() -> None:
    assert SafetyValidator().validate_move(300, 550, True, step=2).allowed


def test_safety_rejects_bounds_stop_and_step_limit() -> None:
    safety = SafetyValidator(max_steps=15)
    assert safety.validate_move(-1, 500, True, step=0).error == "COORDINATES_OUT_OF_BOUNDS"
    assert safety.validate_move(500, 500, True, step=15).error == "MAX_AGENT_STEPS"
    assert safety.validate_move(500, 500, True, step=1, stopped=True).error == "SESSION_STOPPED"

