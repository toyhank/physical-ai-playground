from __future__ import annotations

from typing import Any

from app.simulation.mujoco_engine import MujocoEngine


class MockRobotModel:
    """Deterministic model provider for CI and frontend development."""

    name = "mock"

    def plan(self, prompt: str, engine: MujocoEngine) -> list[dict[str, Any]]:
        normalized = prompt.lower()
        requested_object = "red_cube"
        for keywords, object_id in (
            (("绿色", "green"), "green_cube"),
            (("黄色", "yellow"), "yellow_cube"),
            (("紫色", "purple"), "purple_cube"),
            (("红色", "red"), "red_cube"),
        ):
            if any(keyword in normalized for keyword in keywords):
                requested_object = object_id
                break
        del engine
        return [
            {"name": "pick_object", "arguments": {"object_id": requested_object}},
            {"name": "place_object", "arguments": {"container_id": "blue_box"}},
        ]
