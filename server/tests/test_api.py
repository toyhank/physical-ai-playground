from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_session_lifecycle() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        created = client.post("/api/sessions", json={"seed": 12})
        assert created.status_code == 200
        assert created.json()["model_provider"] in {"mock", "gemini"}
        session_id = created.json()["session_id"]
        state = client.get(f"/api/sessions/{session_id}")
        assert state.status_code == 200
        assert state.json()["state"]["verified"] is False
        camera = client.get(f"/api/sessions/{session_id}/camera.png")
        assert camera.status_code == 200
        assert camera.headers["content-type"] == "image/png"
