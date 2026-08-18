from __future__ import annotations

from typing import Any, Protocol


class RobotModelProvider(Protocol):
    name: str

    def plan(self, prompt: str, engine: Any) -> list[dict[str, Any]]: ...

