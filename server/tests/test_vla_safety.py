import numpy as np

from app.robot.so101_spec import SO101_JOINTS
from app.vla.safety import SafetyFilter


def test_vla_safety_clamps_small_delta_and_rejects_anomaly() -> None:
    safety = SafetyFilter(max_joint_delta=0.1, max_velocity=3.0)
    current = np.asarray(SO101_JOINTS.home, dtype=np.float32)
    result = safety.filter(current + 0.2, current, dt=1 / 30)
    assert result.accepted
    assert "ACTION_CLAMPED" in result.events
    rejected = safety.filter(current + 2.0, current, dt=1 / 30)
    assert not rejected.accepted
    assert rejected.error in {"ACTION_JOINT_LIMIT", "ACTION_DELTA_TOO_LARGE"}


def test_vla_safety_rejects_nan_timeout_and_emergency_stop() -> None:
    safety = SafetyFilter()
    current = np.asarray(SO101_JOINTS.home, dtype=np.float32)
    assert safety.filter(np.full(6, np.nan), current).error == "ACTION_NON_FINITE"
    assert safety.filter(current, current, age_seconds=3).error == "ACTION_TIMEOUT"
    safety.emergency_stop()
    assert safety.filter(current, current).error == "EMERGENCY_STOP"
