# Physical AI Playground

A small, deployable Physical AI demo: type a natural-language task, watch a
robot arm observe the scene, issue `pick_object` / `place_object` skill calls,
and verify the result from simulator state.

The default `mock` provider is deterministic and needs no API key. Set
`MODEL_PROVIDER=gemini` to use `gemini-robotics-er-2-preview`; the browser never
receives the key.

## What is included

- Browser control room with a live scene, camera view, agent trace, run/stop/reset
- FastAPI session service and WebSocket event stream
- Fixed global and eye-in-hand Panda wrist RGB cameras
- Safety-gated semantic skills backed by a stable world-frame object snapshot
- Four independently simulated red, green, yellow, and purple cubes with a new
  collision-safe randomized layout on every reset
- Damped-least-squares arm IK with joint limits
- Google DeepMind MuJoCo Menagerie Franka Panda with its official visual and
  collision meshes, 7 arm actuators, one coupled gripper actuator, `mj_step`
  control, and finger contacts
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
       ├─ safety gateway: pick_object / place_object
       ├─ deterministic skill controller → Cartesian IK → actuators
       └─ MuJoCo engine → dual cameras → verifier → run log
```

## Local quick start

Prerequisites: Node.js 22.13+, Python 3.11+.

```powershell
npm install
python -m venv server/.venv
server/.venv/Scripts/pip.exe install -r server/requirements.txt
```

For local backend settings, create `server/.env.local` (it is ignored by Git):

```dotenv
MODEL_PROVIDER=mock
GEMINI_MODEL=gemini-robotics-er-2-preview
GEMINI_API_KEY=
```

Terminal 1:

```powershell
server/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir server --reload
```

Terminal 2:

```powershell
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:3000`, keep the example prompt, and press **Run task**.

## Gemini Robotics ER 2

Set the provider and key in the ignored `server/.env.local` file:

```dotenv
MODEL_PROVIDER=gemini
GEMINI_MODEL=gemini-robotics-er-2-preview
GEMINI_API_KEY=your-key
MAX_AGENT_STEPS=20
MAX_TASK_SECONDS=240
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
server/.venv/Scripts/python.exe scripts/benchmark_multicolor.py
npm run build
```

The checked-in benchmarks distinguish the lightweight fallback from the native
MuJoCo actuator path. [`benchmarks/mujoco-actuator.json`](benchmarks/mujoco-actuator.json)
records 100/100 successful randomized resets using the Panda's 8 actuators and
`mj_step`.
The multicolor benchmark exercises all four object IDs over 25 randomized
layouts (100 physical pick-and-place trials total).
The model sees both a fixed global RGB view and an RGB view from a camera rigidly
mounted to the Panda hand. It chooses named task-level skills instead of guessing
pixels after the wrist moves. In this simulator, MuJoCo ground truth stands in
for the RGB-D pose estimator that a physical deployment would use to maintain
object poses in the robot base frame. The controller deterministically expands
each skill into approach, descent, grasp/release, lift, transfer, and retreat.
The gripper must first produce a real MuJoCo finger-to-cube contact; only then
does the simulator enable a stabilized attachment for transport.
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
- Native MuJoCo 3.11 actuator benchmark: 100/100 randomized resets
- Official Franka Panda inertial, collision, and visual model (Apache-2.0)
- 7 arm position actuators + 1 coupled gripper actuator; state advances through `mj_step`
- Grasping is contact-gated: no finger contact returns `NO_FINGER_CONTACT`
- Gemini access/spatial/function-call smoke test: passed on the developer account
- 20-run Gemini reliability benchmark is not claimed until run in the target
  billing environment
