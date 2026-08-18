from __future__ import annotations

import asyncio
import base64
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
manager = SessionManager(settings.max_sessions, settings.session_idle_seconds)
orchestrator = AgentOrchestrator(settings)
rate_limits: dict[str, deque[float]] = defaultdict(deque)


class CreateSessionRequest(BaseModel):
    seed: int = 0


class RunTaskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


class ResetRequest(BaseModel):
    seed: int | None = None


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
    try:
        session = manager.create(payload.seed)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"session_id": session.id, "state": session.engine.state()}


@app.get("/api/sessions/{session_id}")
def session_state(session_id: str) -> dict:
    session = get_session(session_id)
    return {"session_id": session.id, "state": session.engine.state()}


@app.get("/api/sessions/{session_id}/camera.png")
def session_camera(session_id: str) -> Response:
    session = get_session(session_id)
    return Response(
        content=base64.b64decode(session.engine.camera_png_base64()),
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
    session.tools = type(session.tools)(session.engine, max_steps=settings.max_agent_steps)
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
    await websocket.send_json({"type": "scene_state", "state": session.engine.state()})
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)
