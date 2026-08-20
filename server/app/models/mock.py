from __future__ import annotations

from typing import Any

from app.simulation.mujoco_engine import MujocoEngine


class MockRobotModel:
    """Deterministic model provider for CI and frontend development."""

    name = "mock"

    def plan(self, prompt: str, engine: MujocoEngine) -> list[dict[str, Any]]:
        del prompt  # All supported languages use the same visual task loop.
        del engine
        return [
            {"name": "pick_object", "arguments": {"object_id": "red_cube"}},
            {"name": "place_object", "arguments": {"container_id": "blue_box"}},
        ]
