from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


LOCAL_ENV_PATH = Path(__file__).resolve().parents[1] / ".env.local"
load_dotenv(LOCAL_ENV_PATH, override=False)


@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "mock")
    model_id: str = os.getenv("GEMINI_MODEL", "gemini-robotics-er-2-preview")
    vla_provider: str = os.getenv("VLA_PROVIDER", "mock")
    vla_policy_path: str = os.getenv(
        "VLA_POLICY_PATH", "ahmedsohail2003/smolvla-so101-pickplace-v2"
    )
    vla_task_prompt: str = os.getenv(
        "VLA_TASK_PROMPT", "Pick up the red block and place it in the blue tray."
    )
    vla_device: str = os.getenv("VLA_DEVICE", "cuda")
    vla_host: str = os.getenv("VLA_HOST", "http://127.0.0.1:8100")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "15"))
    vla_max_steps: int = int(os.getenv("VLA_MAX_STEPS", "240"))
    max_task_seconds: int = int(os.getenv("MAX_TASK_SECONDS", "60"))
    session_idle_seconds: int = int(os.getenv("SESSION_IDLE_SECONDS", "120"))
    max_sessions: int = int(os.getenv("MAX_SESSIONS", "4"))
    tasks_per_hour: int = int(os.getenv("TASKS_PER_HOUR", "10"))
    public_vla_only: bool = os.getenv("PUBLIC_VLA_ONLY", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    public_allow_gemini: bool = os.getenv(
        "PUBLIC_ALLOW_GEMINI", "false"
    ).lower() in {"1", "true", "yes"}
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    )


settings = Settings()
