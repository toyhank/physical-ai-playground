from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.agent.tools import RobotTools
from app.robot.backends import PandaMujocoBackend, SO101MujocoBackend


ROBOT_BACKENDS = {
    "panda": PandaMujocoBackend,
    "so101": SO101MujocoBackend,
}


class SimulationSession:
    def __init__(
        self,
        seed: int = 0,
        *,
        robot: str = "panda",
        controller: str = "classical",
        brain: str = "none",
        policy: str = "mock_vla",
        grasp_mode: str = "physics",
    ) -> None:
        self.id = uuid4().hex
        backend_type = ROBOT_BACKENDS[robot]
        if robot == "panda":
            self.backend = backend_type(seed=seed, enable_mujoco=True)
        else:
            self.backend = backend_type(
                seed=seed,
                grasp_mode=grasp_mode,
                vla_aligned=controller in {"vla", "hybrid"},
            )
        self.engine = self.backend  # Compatibility alias for existing clients/tests.
        self.robot = robot
        self.controller = controller
        self.brain = brain
        self.policy = policy
        self.grasp_mode = grasp_mode
        self.tools = RobotTools(self.backend)
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.running_task: asyncio.Task[None] | None = None
        self.last_active = datetime.now(UTC)

    def touch(self) -> None:
        self.last_active = datetime.now(UTC)

    async def publish(self, payload: dict[str, Any]) -> None:
        self.touch()
        for queue in tuple(self.subscribers):
            await queue.put(payload)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self.subscribers.add(queue)
        self.touch()
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.subscribers.discard(queue)

    def stop(self) -> None:
        self.tools.stop()
        self.backend.stop()
        if self.running_task and not self.running_task.done():
            self.running_task.cancel()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.stop()
        self.tools = RobotTools(self.backend)
        self.touch()
        return self.backend.reset(seed)
