from __future__ import annotations

import asyncio
import base64
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from app.agent.orchestrator import AgentOrchestrator
from app.config import settings
from app.sessions.manager import SessionManager


app = FastAPI(title="Physical AI Playground API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/models",
    StaticFiles(directory=Path(__file__).resolve().parents[2] / "public" / "models"),
    name="models",
)
manager = SessionManager(settings.max_sessions, settings.session_idle_seconds)
orchestrator = AgentOrchestrator(settings)
rate_limits: dict[str, deque[float]] = defaultdict(deque)


class CreateSessionRequest(BaseModel):
    seed: int = 0
    robot: Literal["so101", "panda"] = "so101"
    controller: Literal["classical", "vla", "hybrid"] = "vla"
    brain: Literal["none", "gemini"] = "none"
    policy: Literal["smolvla", "mock_vla"] = "smolvla"
    grasp_mode: Literal["physics", "contact_attachment"] = "physics"

    @model_validator(mode="after")
    def validate_configuration(self) -> "CreateSessionRequest":
        if self.robot == "panda" and self.controller != "classical":
            raise ValueError("PANDA_ONLY_SUPPORTS_CLASSICAL")
        if self.controller == "hybrid" and self.brain != "gemini":
            raise ValueError("HYBRID_REQUIRES_GEMINI_BRAIN")
        return self


class RunTaskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


class ResetRequest(BaseModel):
    seed: int | None = None


def public_configuration_allowed(
    payload: CreateSessionRequest, *, allow_gemini: bool
) -> bool:
    common = (
        payload.robot == "so101"
        and payload.policy == "smolvla"
        and payload.grasp_mode == "physics"
    )
    vla = payload.controller == "vla" and payload.brain == "none"
    hybrid = (
        allow_gemini
        and payload.controller == "hybrid"
        and payload.brain == "gemini"
    )
    return common and (vla or hybrid)


def get_session(session_id: str):
    try:
        return manager.get(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND") from error


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_provider": settings.model_provider,
        "active_sessions": len(manager.sessions),
    }


@app.post("/api/sessions")
def create_session(payload: CreateSessionRequest) -> dict:
    if settings.public_vla_only and not public_configuration_allowed(
        payload, allow_gemini=settings.public_allow_gemini
    ):
        raise HTTPException(status_code=403, detail="PUBLIC_DEMO_VLA_ONLY")
    try:
        session = manager.create(
            payload.seed,
            robot=payload.robot,
            controller=payload.controller,
            brain=payload.brain,
            policy=payload.policy,
            grasp_mode=payload.grasp_mode,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "session_id": session.id,
        "state": session.backend.get_state(),
        "model_provider": settings.model_provider,
        "configuration": {
            "robot": session.robot,
            "controller": session.controller,
            "brain": session.brain,
            "policy": session.policy,
            "grasp_mode": session.grasp_mode,
        },
    }


@app.get("/api/sessions/{session_id}")
def session_state(session_id: str) -> dict:
    session = get_session(session_id)
    return {"session_id": session.id, "state": session.backend.get_state()}


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> Response:
    try:
        manager.delete(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND") from error
    return Response(status_code=204)


@app.get("/api/sessions/{session_id}/camera.png")
def session_camera(session_id: str, view: Literal["scene", "wrist"] = "scene") -> Response:
    session = get_session(session_id)
    return Response(
        content=base64.b64decode(
            session.backend.render_camera(view, latch_observation=False)
        ),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/sessions/{session_id}/run", status_code=202)
async def run_task(session_id: str, payload: RunTaskRequest, request: Request) -> dict:
    session = get_session(session_id)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    history = rate_limits[client_ip]
    while history and now - history[0] > 3600:
        history.popleft()
    if len(history) >= settings.tasks_per_hour:
        raise HTTPException(status_code=429, detail="TASK_RATE_LIMIT")
    if session.running_task and not session.running_task.done():
        raise HTTPException(status_code=409, detail="TASK_ALREADY_RUNNING")
    history.append(now)
    session.tools = type(session.tools)(session.backend, max_steps=settings.max_agent_steps)
    session.running_task = asyncio.create_task(orchestrator.run(session, payload.prompt))
    return {"accepted": True, "session_id": session.id}


@app.post("/api/sessions/{session_id}/stop")
async def stop_task(session_id: str) -> dict:
    session = get_session(session_id)
    session.stop()
    return {"stopped": True}


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str, payload: ResetRequest) -> dict:
    session = get_session(session_id)
    state = session.reset(payload.seed)
    await session.publish({"type": "scene_state", "state": state})
    return {"state": state}


@app.websocket("/ws/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str) -> None:
    try:
        session = manager.get(session_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    queue = session.subscribe()
    await websocket.send_json({"type": "scene_state", "state": session.backend.get_state()})
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)
