from __future__ import annotations

import numpy as np
import pytest

from app.simulation.mujoco_engine import MujocoEngine


def test_mujoco_uses_panda_actuators_and_steps_dynamics() -> None:
    engine = MujocoEngine(seed=7, enable_mujoco=True)
    if not engine.mujoco_enabled:
        pytest.skip("MuJoCo is not installed in this environment")

    assert engine.model.nu == 8
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
    cube_x, cube_y = engine.world_to_camera_normalized(engine.cube_position)
    assert engine.move(cube_x, cube_y, high=True).success
    assert engine.move(cube_x, cube_y, high=False).success

    result = engine.set_gripper_state(False)
    closed = engine.data.qpos[7:9].copy()

    assert result.success
    assert result.as_dict()["contact_count"] >= 1
    assert result.as_dict()["grasp_constraint"] == "contact-gated-attachment"
    assert engine.grasped
    assert np.all(closed < opened)
    assert np.isclose(engine.data.ctrl[7], 0.0)


def test_robot_camera_is_mounted_on_the_moving_hand() -> None:
    engine = MujocoEngine(seed=9, enable_mujoco=True)
    if not engine.mujoco_enabled:
        pytest.skip("MuJoCo is not installed in this environment")

    camera_id = engine._camera_id
    assert camera_id is not None
    hand_id = engine._mujoco.mj_name2id(
        engine.model, engine._mujoco.mjtObj.mjOBJ_BODY, "hand"
    )
    assert int(engine.model.cam_bodyid[camera_id]) == hand_id

    initial_position = engine.data.cam_xpos[camera_id].copy()
    target = engine.robot_joints.copy()
    target[0] += 0.2
    engine._step_actuators(target, substeps=80)

    assert not np.allclose(engine.data.cam_xpos[camera_id], initial_position)


def test_observation_rays_resolve_cube_and_container_geometry() -> None:
    engine = MujocoEngine(seed=23, enable_mujoco=True)
    if not engine.mujoco_enabled:
        pytest.skip("MuJoCo is not installed in this environment")

    cube_x, cube_y = engine.world_to_camera_normalized(engine.cube_position)
    box_x, box_y = engine.world_to_camera_normalized(engine.box_position)

    assert engine._observation_ray_target(cube_x, cube_y) == "cube"
    assert engine._observation_ray_target(box_x, box_y) == "box"
