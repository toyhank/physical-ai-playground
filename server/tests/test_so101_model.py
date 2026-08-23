from app.robot.backends import SO101MujocoBackend
from app.robot.so101_spec import SO101_JOINTS


def test_so101_model_loads_with_six_named_actuators() -> None:
    backend = SO101MujocoBackend(seed=0)
    assert backend.model.nu == 6
    assert tuple(backend._actuator_ids) == SO101_JOINTS.names
    assert backend.joint_positions.shape == (6,)
    assert backend.model.opt.timestep == 0.002
    backend.stop()


def test_so101_cameras_render_rgb_png() -> None:
    backend = SO101MujocoBackend(seed=0)
    assert len(backend.render_camera("scene")) > 1000
    assert len(backend.render_camera("wrist")) > 1000
    backend.stop()
