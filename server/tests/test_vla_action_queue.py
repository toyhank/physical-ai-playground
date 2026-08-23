import numpy as np
import pytest

from app.vla.action_queue import ActionQueue


def test_action_queue_refills_before_it_runs_empty() -> None:
    queue = ActionQueue(refill_threshold=2)
    queue.extend(np.zeros((5, 6), dtype=np.float32))
    assert not queue.needs_refill
    queue.pop()
    queue.pop()
    queue.pop()
    assert queue.needs_refill
    queue.clear()
    with pytest.raises(IndexError, match="ACTION_QUEUE_EMPTY"):
        queue.pop()
