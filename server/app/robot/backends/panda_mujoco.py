from __future__ import annotations

from typing import Any

from app.robot.base import RobotBackend
from app.simulation.mujoco_engine import MujocoEngine, ToolExecution


class PandaMujocoBackend(RobotBackend):
    """Legacy Franka Panda adapter kept as a reference backend."""

    robot_id = "panda"

    def __init__(self, seed: int = 0, *, enable_mujoco: bool = True) -> None:
        self.legacy = MujocoEngine(seed=seed, enable_mujoco=enable_mujoco)
        self._stopped = False

    def __getattr__(self, name: str) -> Any:
        # Preserve compatibility for diagnostics while upper layers migrate to
        # the RobotBackend contract.
        return getattr(self.legacy, name)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._stopped = False
        return self.legacy.reset(seed)

    def get_state(self) -> dict[str, Any]:
        return self.legacy.state()

    def get_observation(self) -> dict[str, Any]:
        return {
            "observation.images.scene": self.render_camera(
                "scene", latch_observation=True
            ),
            "observation.images.wrist": self.render_camera("wrist"),
            "observation.state": self.legacy.robot_joints.astype("float32").tolist(),
        }

    def apply_action(self, action: dict[str, Any]) -> ToolExecution:
        name = action.get("name")
        arguments = action.get("arguments", {})
        if name == "move":
            return self.legacy.move(**arguments)
        if name == "set_gripper_state":
            return self.legacy.set_gripper_state(**arguments)
        if name == "pick_object":
            return self.legacy.pick_object(**arguments)
        if name == "place_object":
            return self.legacy.place_object(**arguments)
        return ToolExecution(False, "UNKNOWN_ACTION")

    def render_camera(self, camera: str, *, latch_observation: bool = False) -> str:
        return self.legacy.camera_png_base64(
            camera=camera, latch_observation=latch_observation
        )

    def verify_task(self) -> bool:
        return self.legacy.verify_task()

    def stop(self) -> None:
        self._stopped = True

