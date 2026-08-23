from app.robot.backends import SO101MujocoBackend
from app.vla.controller import VLAController
from app.vla.mock import MockVLAProvider


def test_mock_vla_completes_observation_action_physics_loop() -> None:
    backend = SO101MujocoBackend(seed=0)
    controller = VLAController(backend, MockVLAProvider(chunk_size=8))
    before = backend.simulation_steps
    ticks = [controller.tick("Move smoothly for an integration smoke test.") for _ in range(5)]
    assert all(tick.result.success for tick in ticks)
    assert backend.simulation_steps > before
    assert all(len(tick.raw_action) == 6 for tick in ticks)
    assert all(tick.safe_action is not None for tick in ticks)
    backend.stop()


def test_mock_vla_rejects_ground_truth_observation() -> None:
    backend = SO101MujocoBackend(seed=0)
    observation = backend.get_observation()
    observation["object_xyz"] = [0, 0, 0]
    try:
        MockVLAProvider().infer(observation, "test")
    except ValueError as error:
        assert str(error) == "VLA_OBSERVATION_CONTAINS_FORBIDDEN_FEATURES"
    else:
        raise AssertionError("forbidden ground-truth feature was accepted")
    backend.stop()
