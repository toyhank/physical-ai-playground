from app.robot.backends import SO101MujocoBackend


def test_physics_grasp_reports_both_jaw_contacts() -> None:
    backend = SO101MujocoBackend(seed=0, grasp_mode="physics")
    result = backend.pick_object("red_cube")
    assert result.success
    metrics = backend.get_state()["physics_metrics"]
    assert metrics["left_contact"]
    assert metrics["right_contact"]
    assert metrics["normal_force"] > 0
    assert metrics["object_height"] > 0.05
    backend.stop()
