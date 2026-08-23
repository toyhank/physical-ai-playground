from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class VLAProvider(Protocol):
    name: str

    def infer(self, observation: dict[str, Any], task: str) -> np.ndarray: ...

    def reset_session(self) -> None: ...
