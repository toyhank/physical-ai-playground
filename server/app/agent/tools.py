from __future__ import annotations

from typing import Any

from app.robot.safety import SafetyValidator
from app.simulation.mujoco_engine import MujocoEngine, ToolExecution


class RobotTools:
    def __init__(self, engine: MujocoEngine, max_steps: int = 15) -> None:
        self.engine = engine
        self.safety = SafetyValidator(max_steps=max_steps)
        self.steps = 0
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        if name == "move":
            validation = self.safety.validate_move(
                arguments.get("x"), arguments.get("y"), arguments.get("high"), self.steps, self.stopped
            )
        elif name == "set_gripper_state":
            validation = self.safety.validate_gripper(arguments.get("opened"), self.steps, self.stopped)
        elif name == "pick_object":
            validation = self.safety.validate_semantic_skill(
                arguments.get("object_id"), {"red_cube"}, self.steps, self.stopped
            )
        elif name == "place_object":
            validation = self.safety.validate_semantic_skill(
                arguments.get("container_id"), {"blue_box"}, self.steps, self.stopped
            )
        else:
            return ToolExecution(False, "UNKNOWN_TOOL")
        if not validation.allowed:
            return ToolExecution(False, validation.error)
        self.steps += 1
        if name == "move":
            return self.engine.move(arguments["x"], arguments["y"], arguments["high"])
        if name == "set_gripper_state":
            return self.engine.set_gripper_state(arguments["opened"])
        if name == "pick_object":
            return self.engine.pick_object(arguments["object_id"])
        return self.engine.place_object(arguments["container_id"])
