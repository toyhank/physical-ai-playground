from __future__ import annotations

from datetime import UTC, datetime

from app.sessions.session import SimulationSession


class SessionManager:
    def __init__(self, max_sessions: int = 4, idle_seconds: int = 120) -> None:
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self.sessions: dict[str, SimulationSession] = {}

    def create(self, seed: int = 0, **options: str) -> SimulationSession:
        self.cleanup()
        if len(self.sessions) >= self.max_sessions:
            disconnected = sorted(
                (
                    session
                    for session in self.sessions.values()
                    if not session.subscribers
                    and (
                        session.running_task is None
                        or session.running_task.done()
                    )
                ),
                key=lambda session: session.last_active,
            )
            if disconnected:
                stale = disconnected[0]
                stale.stop()
                del self.sessions[stale.id]
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("SESSION_CAPACITY_REACHED")
        session = SimulationSession(seed=seed, **options)
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> SimulationSession:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(session_id)
        session.touch()
        return session

    def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            raise KeyError(session_id)
        session.stop()

    def cleanup(self) -> None:
        now = datetime.now(UTC)
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if (now - session.last_active).total_seconds() > self.idle_seconds
        ]
        for session_id in expired:
            self.sessions[session_id].stop()
            del self.sessions[session_id]
