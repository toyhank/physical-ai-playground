from __future__ import annotations

import numpy as np
import pytest

from app.simulation.mujoco_engine import MujocoEngine


def test_mujoco_uses_nine_position_actuators_and_steps_dynamics() -> None:
    engine = MujocoEngine(seed=7, enable_mujoco=True)
    if not engine.mujoco_enabled:
        pytest.skip("MuJoCo is not installed in this environment")

    assert engine.model.nu == 9
    assert engine.state()["control_mode"] == "actuator-mj_step"
    initial_time = float(engine.data.time)
    initial_qpos = engine.data.qpos[:7].copy()
    target = initial_qpos.copy()
    target[0] += 0.25

    engine._step_actuators(target, substeps=80)

    assert engine.data.time > initial_time
    assert engine.simulation_steps == 80
    assert not np.allclose(engine.data.qpos[:7], initial_qpos)
    assert np.allclose(engine.robot_joints, engine.data.qpos[:7])
    assert np.isclose(engine.data.ctrl[0], target[0])


def test_gripper_finger_joints_are_actuated() -> None:
    engine = MujocoEngine(seed=8, enable_mujoco=True)
    if not engine.mujoco_enabled:
        pytest.skip("MuJoCo is not installed in this environment")

    opened = engine.data.qpos[7:9].copy()
    assert np.all(opened > 0.035)
    engine.set_gripper_state(False)
    closed = engine.data.qpos[7:9].copy()

    assert np.all(closed < opened)
    assert np.allclose(engine.data.ctrl[7:9], 0.0)
