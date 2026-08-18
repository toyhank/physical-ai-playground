from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "mock")
    model_id: str = os.getenv("GEMINI_MODEL", "gemini-robotics-er-2-preview")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "15"))
    max_task_seconds: int = int(os.getenv("MAX_TASK_SECONDS", "60"))
    session_idle_seconds: int = int(os.getenv("SESSION_IDLE_SECONDS", "120"))
    max_sessions: int = int(os.getenv("MAX_SESSIONS", "4"))
    tasks_per_hour: int = int(os.getenv("TASKS_PER_HOUR", "10"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    )


settings = Settings()
