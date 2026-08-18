from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.agent.tools import RobotTools
from app.simulation.mujoco_engine import MujocoEngine


class SimulationSession:
    def __init__(self, seed: int = 0) -> None:
        self.id = uuid4().hex
        self.engine = MujocoEngine(seed=seed, enable_mujoco=True)
        self.tools = RobotTools(self.engine)
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
        if self.running_task and not self.running_task.done():
            self.running_task.cancel()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.stop()
        self.tools = RobotTools(self.engine)
        self.touch()
        return self.engine.reset(seed)

