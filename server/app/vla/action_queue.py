from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np


class ActionQueue:
    def __init__(self, refill_threshold: int = 4) -> None:
        if refill_threshold < 0:
            raise ValueError("INVALID_REFILL_THRESHOLD")
        self.refill_threshold = refill_threshold
        self._items: deque[np.ndarray] = deque()

    def extend(self, chunk: Iterable[np.ndarray | list[float]]) -> None:
        for action in chunk:
            array = np.asarray(action, dtype=np.float32)
            if array.shape != (6,):
                raise ValueError("ACTION_SHAPE_MISMATCH")
            self._items.append(array)

    def pop(self) -> np.ndarray:
        if not self._items:
            raise IndexError("ACTION_QUEUE_EMPTY")
        return self._items.popleft()

    @property
    def needs_refill(self) -> bool:
        return len(self._items) <= self.refill_threshold

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
