from __future__ import annotations

from datetime import UTC, datetime

import pytest

import app.sessions.manager as manager_module
from app.sessions.manager import SessionManager


class FakeSession:
    next_id = 0

    def __init__(self, seed: int = 0, **options: str) -> None:
        type(self).next_id += 1
        self.id = f"session-{type(self).next_id}"
        self.subscribers: set[object] = set()
        self.running_task = None
        self.last_active = datetime.now(UTC)
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_capacity_recycles_disconnected_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "SimulationSession", FakeSession)
    manager = SessionManager(max_sessions=1, idle_seconds=120)
    first = manager.create(seed=1)

    second = manager.create(seed=2)

    assert first.stopped is True
    assert list(manager.sessions) == [second.id]


def test_capacity_preserves_connected_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "SimulationSession", FakeSession)
    manager = SessionManager(max_sessions=1, idle_seconds=120)
    first = manager.create(seed=1)
    first.subscribers.add(object())

    with pytest.raises(RuntimeError, match="SESSION_CAPACITY_REACHED"):
        manager.create(seed=2)

    assert first.stopped is False
