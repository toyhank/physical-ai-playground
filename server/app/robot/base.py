from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RobotBackend(ABC):
    """Robot-independent boundary used by sessions, agents, and policies."""

    robot_id: str

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_observation(self) -> dict[str, Any]: ...

    @abstractmethod
    def apply_action(self, action: dict[str, Any]) -> Any: ...

    @abstractmethod
    def render_camera(self, camera: str, *, latch_observation: bool = False) -> str: ...

    @abstractmethod
    def verify_task(self) -> bool: ...

    @abstractmethod
    def stop(self) -> None: ...

