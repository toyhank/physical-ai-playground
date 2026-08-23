import numpy as np

from app.robot.backends import SO101MujocoBackend
from app.robot.so101_spec import SO101_JOINTS


def test_so101_joint_action_and_limits() -> None:
    backend = SO101MujocoBackend(seed=0)
    target = backend.joint_positions.copy()
    target[0] += 0.05
    assert backend.apply_action({"name": "joint_position", "arguments": {"values": target}}).success
    invalid = np.asarray(SO101_JOINTS.upper) + 1.0
    result = backend.apply_action({"name": "joint_position", "arguments": {"values": invalid}})
    assert not result.success
    assert result.error == "JOINT_LIMIT"
    backend.stop()


def test_policy_frequency_uses_16_or_17_physics_steps() -> None:
    backend = SO101MujocoBackend(seed=0)
    before = backend.simulation_steps
    backend.apply_action(
        {"name": "joint_position", "arguments": {"values": backend.joint_positions}}
    )
    assert backend.simulation_steps - before in {16, 17}
    backend.stop()
