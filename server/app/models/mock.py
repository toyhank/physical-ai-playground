from __future__ import annotations

from typing import Any

from app.simulation.mujoco_engine import MujocoEngine


class MockRobotModel:
    """Deterministic model provider for CI and frontend development."""

    name = "mock"

    def plan(self, prompt: str, engine: MujocoEngine) -> list[dict[str, Any]]:
        del prompt  # All supported languages use the same visual task loop.
        cube_x, cube_y = engine.world_to_camera_normalized(engine.cube_position)
        box_x, box_y = engine.world_to_camera_normalized(engine.box_position)
        return [
            {"name": "set_gripper_state", "arguments": {"opened": True}},
            {"name": "move", "arguments": {"x": cube_x, "y": cube_y, "high": True}},
            {"name": "move", "arguments": {"x": cube_x, "y": cube_y, "high": False}},
            {"name": "set_gripper_state", "arguments": {"opened": False}},
            {"name": "move", "arguments": {"x": cube_x, "y": cube_y, "high": True}},
            {"name": "move", "arguments": {"x": box_x, "y": box_y, "high": True}},
            {"name": "move", "arguments": {"x": box_x, "y": box_y, "high": False}},
            {"name": "set_gripper_state", "arguments": {"opened": True}},
            {"name": "move", "arguments": {"x": box_x, "y": box_y, "high": True}},
        ]
