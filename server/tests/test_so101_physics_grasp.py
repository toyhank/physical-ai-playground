from app.robot.backends import SO101MujocoBackend


def test_physics_mode_lifts_without_attachment() -> None:
    backend = SO101MujocoBackend(seed=0, grasp_mode="physics")
    assert backend._attached_object is None
    result = backend.pick_object("red_cube")
    assert result.success
    assert backend._attached_object is None
    assert backend.get_state()["physics_metrics"]["object_height"] > 0.05
    backend.stop()


def test_cube_falls_under_gravity() -> None:
    backend = SO101MujocoBackend(seed=0)
    address = backend._cube_joint_addresses["red_cube"]
    backend.data.qpos[address + 2] = 0.15
    backend._mujoco.mj_forward(backend.model, backend.data)
    initial = float(backend.data.qpos[address + 2])
    for _ in range(100):
        backend._mujoco.mj_step(backend.model, backend.data)
    assert float(backend.data.qpos[address + 2]) < initial
    backend.stop()


def test_vla_verification_requires_cube_fully_inside_stable_and_released() -> None:
    backend = SO101MujocoBackend(seed=0, vla_aligned=True)
    center = backend._box_position.copy()
    center[2] += backend.cube_half_size
    backend._write_cube_pose("red_cube", center)
    backend.data.qpos[backend._joint_qpos["gripper"]] = 1.0
    backend._mujoco.mj_forward(backend.model, backend.data)
    assert backend.verify_task()

    partly_over_wall = center.copy()
    partly_over_wall[0] += 0.03
    backend._write_cube_pose("red_cube", partly_over_wall)
    backend._mujoco.mj_forward(backend.model, backend.data)
    assert not backend.verify_task()

    backend._write_cube_pose("red_cube", center)
    backend.data.qpos[backend._joint_qpos["gripper"]] = 0.0
    backend._mujoco.mj_forward(backend.model, backend.data)
    assert not backend.verify_task()
    backend.stop()
