from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.robot.backends.so101_mujoco import SO101MujocoBackend
from app.simulation.mujoco_engine import ToolExecution
from app.vla.action_codec import ActionCodec
from app.vla.action_queue import ActionQueue
from app.vla.base import VLAProvider
from app.vla.safety import SafetyFilter


@dataclass
class VLATick:
    result: ToolExecution
    raw_action: list[float]
    safe_action: list[float] | None
    safety_events: list[str]
    queue_size: int


class VLAController:
    def __init__(
        self,
        backend: SO101MujocoBackend,
        provider: VLAProvider,
        *,
        codec: ActionCodec | None = None,
        safety: SafetyFilter | None = None,
        refill_threshold: int = 0,
    ) -> None:
        self.backend = backend
        self.provider = provider
        self.codec = codec or ActionCodec()
        self.safety = safety or (
            SafetyFilter(max_joint_delta=float("inf"), max_velocity=float("inf"))
            if backend.vla_aligned
            else SafetyFilter()
        )
        self.queue = ActionQueue(refill_threshold)
        self.last_inference_ms = 0.0

    def refill(self, task: str) -> None:
        import time

        observation = self.backend.get_observation()
        started = time.perf_counter()
        chunk = self.provider.infer(observation, task)
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        decoded = [self.codec.decode_policy_action(action) for action in chunk]
        self.queue.extend(decoded)

    def tick(self, task: str) -> VLATick:
        if self.queue.needs_refill:
            self.refill(task)
        raw = self.queue.pop()
        safe = self.safety.filter(
            raw,
            self.backend.joint_positions,
            dt=1 / self.backend.policy_hz,
        )
        if not safe.accepted or safe.action is None:
            result = ToolExecution(False, safe.error)
            safe_action = None
        else:
            result = self.backend.apply_action(
                {"name": "joint_position", "arguments": {"values": safe.action}}
            )
            safe_action = safe.action.astype(float).tolist()
        return VLATick(
            result=result,
            raw_action=raw.astype(float).tolist(),
            safe_action=safe_action,
            safety_events=safe.events,
            queue_size=len(self.queue),
        )

    def reset(self) -> None:
        self.queue.clear()
        self.safety.reset()
        self.provider.reset_session()
