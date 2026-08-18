# Physical AI Playground

A small, deployable Physical AI demo: type a natural-language task, watch a
robot arm observe the scene, issue `move` / `set_gripper_state` function calls,
and verify the result from simulator state.

The default `mock` provider is deterministic and needs no API key. Set
`MODEL_PROVIDER=gemini` to use `gemini-robotics-er-2-preview`; the browser never
receives the key.

## What is included

- Browser control room with a live scene, camera view, agent trace, run/stop/reset
- FastAPI session service and WebSocket event stream
- Safety-gated normalized image-coordinate tools
- Pinhole camera projection and table-plane unprojection
- Damped-least-squares arm IK with joint limits
- MuJoCo MJCF scene, plus a deterministic kinematic fallback for lightweight dev
- Simulator-ground-truth task verification and JSONL run logs
- Mock and Gemini Robotics ER 2 providers behind the same orchestration interface
- Unit, integration, randomized pick-and-place, and opt-in external API tests

## Architecture

```text
Browser (vinext/React)
  ├─ POST /api/sessions, /run, /stop, /reset
  └─ WebSocket /ws/{session_id}
                │
FastAPI session manager
  └─ Agent orchestrator
       ├─ mock provider OR Gemini Robotics ER 2
       ├─ safety gateway: move / set_gripper_state
       └─ MuJoCo engine → camera → verifier → run log
```

## Local quick start

Prerequisites: Node.js 22.13+, Python 3.11+.

```powershell
npm install
python -m venv server/.venv
server/.venv/Scripts/pip.exe install -r server/requirements.txt
```

Terminal 1:

```powershell
$env:MODEL_PROVIDER="mock"
server/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir server --reload
```

Terminal 2:

```powershell
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:3000`, keep the example prompt, and press **Run task**.

## Gemini Robotics ER 2

Set the key only in your local shell or deployment secret manager:

```powershell
$env:GEMINI_API_KEY="..."
$env:MODEL_PROVIDER="gemini"
```

Never commit the key or put it in a `NEXT_PUBLIC_*` variable. Run the standalone
smoke test before enabling the live provider:

```powershell
server/.venv/Scripts/python.exe scripts/test_gemini_robotics.py
```

The script draws its own scene, checks `[y,x]` spatial coordinates, and mocks all
tool results. It does not connect to robot hardware.

## Tests and benchmark

```powershell
server/.venv/Scripts/python.exe -m pytest
server/.venv/Scripts/python.exe scripts/benchmark_mock.py
npm run build
```

The checked-in local fallback benchmark is
[`benchmarks/deterministic-fallback.json`](benchmarks/deterministic-fallback.json):
100/100 successful randomized resets. It is explicitly not a MuJoCo result.
Re-run the same benchmark in the container to validate the installed MuJoCo path.
External Gemini tests are opt-in:

```powershell
$env:RUN_GEMINI_TESTS="1"
server/.venv/Scripts/python.exe -m pytest -m gemini
```

## Docker backend

```powershell
docker compose up --build
```

The image installs the official `mujoco` Python package and headless OSMesa/EGL
libraries. Persisted logs appear under `data/runs/`.

For a public frontend, set `NEXT_PUBLIC_API_URL` to the HTTPS backend URL before
building. Set backend `CORS_ORIGINS` to the deployed frontend origin. The API
already enforces 2–4 active sessions, idle cleanup, max agent steps, timeout,
stop, and per-IP task rate limits.

## Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PROVIDER` | `mock` | `mock` or `gemini` |
| `GEMINI_MODEL` | `gemini-robotics-er-2-preview` | Robotics model ID |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Browser-visible API base |
| `CORS_ORIGINS` | local frontend origins | Comma-separated allowed origins |
| `MAX_AGENT_STEPS` | `15` | Tool-call safety ceiling |
| `MAX_TASK_SECONDS` | `60` | Per-task timeout |
| `MAX_SESSIONS` | `4` | Concurrent simulation sessions |
| `TASKS_PER_HOUR` | `10` | Per-IP rate limit |

## Current validation status

- Local UI build and real HTTP/WebSocket mock task: passed
- Projection, IK, safety, orchestration, API lifecycle: passed
- Randomized deterministic fallback: 100/100
- Gemini access/spatial/function-call smoke test: passed on the developer account
- MuJoCo benchmark and 20-run Gemini reliability benchmark: scripts/interfaces are
  ready, but those results are not claimed until run in the target container and
  billing environment
