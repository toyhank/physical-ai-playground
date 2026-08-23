from __future__ import annotations

from app.robot.backends import PandaMujocoBackend, SO101MujocoBackend
from app.robot.base import RobotBackend


def test_panda_implements_robot_backend_contract() -> None:
    backend: RobotBackend = PandaMujocoBackend(seed=5, enable_mujoco=False)
    state = backend.get_state()
    observation = backend.get_observation()

    assert backend.robot_id == "panda"
    assert state["physics"] == "kinematic-fallback"
    assert set(observation) == {
        "observation.images.scene",
        "observation.images.wrist",
        "observation.state",
    }
    assert len(observation["observation.state"]) == 7
    assert backend.apply_action({"name": "unknown", "arguments": {}}).error == "UNKNOWN_ACTION"


def test_so101_vla_reference_matches_checkpoint_environment() -> None:
    backend = SO101MujocoBackend(seed=123, vla_aligned=True)
    state = backend.get_state()

    assert state["scene_profile"] == "vla_reference"
    assert state["policy_fps"] == 15
    assert [item["name"] for item in state["objects"]] == ["red_cube", "blue_box"]
    assert state["objects"][1]["position"] == [-0.18, -0.13, 0.006]
    assert backend._physics_substeps_for_action() == 33
    backend.stop()
