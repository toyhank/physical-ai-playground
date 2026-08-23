# Physical AI Playground

A browser-based robotics lab for comparing three control architectures on the
same MuJoCo workcell:

- **Classical** — Gemini ER2/mock → semantic skill → ground-truth pose → IK → SO-101
- **VLA** — scene RGB + wrist RGB + 6D joint state + language → SmolVLA → safety → SO-101
- **ER2 + VLA** — Gemini Robotics ER2 decomposes a task into language subtasks; SmolVLA executes them

The default robot is the official upstream **SO-101** model. The previous
Franka Panda implementation remains available as **Panda Legacy**.

## Architecture

```text
                              ┌─ Classical expert ─ ground truth + IK ─┐
Browser ─ FastAPI session ────┼─ SmolVLA service ─ ActionCodec ───────┼─ SafetyFilter
  │ RGB/state/trace           └─ Gemini ER2 ─ language VLA subtask ───┘       │
  │                                                                            ▼
  └──────── Three.js rigid-body transforms ◀── SO-101 RobotBackend ◀── ActionQueue @ 30 Hz
                                                   │
                                  MuJoCo contact physics @ 500 Hz
                                  scene camera + wrist camera
                                  task verifier + physics diagnostics
```

The VLA boundary is deliberately narrow. Runtime and training policy features
are only:

```text
observation.images.scene
observation.images.wrist
observation.state       float32[6]
task                    natural language
```

Object XYZ, MuJoCo ground truth, target IDs, segmentation masks and IK targets
are never passed into the VLA controller. Ground truth is reserved for the
classical baseline, demonstration generation, regression tests and final task
verification.

## Local quick start

Prerequisites: Node.js 22+, Python 3.11–3.13 for the playground. Use Python
3.12/3.13 for the optional LeRobot/SmolVLA environment.

```powershell
npm install
python -m venv server/.venv
server/.venv/Scripts/pip.exe install -r server/requirements.txt
```

Create ignored `server/.env.local`:

```dotenv
MODEL_PROVIDER=mock
GEMINI_MODEL=gemini-robotics-er-2-preview
GEMINI_API_KEY=

VLA_PROVIDER=mock
VLA_POLICY_PATH=ahmedsohail2003/smolvla-so101-pickplace-v2
VLA_TASK_PROMPT=Pick up the red block and place it in the blue tray.
VLA_POLICY_JOINT_OFFSETS=0,1.5707963268,-1.5707963268,0,1.5707963268,0
VLA_DEVICE=cuda
VLA_HOST=http://127.0.0.1:8100
```

Start the backend and frontend:

```powershell
# terminal 1
server/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir server --reload

# terminal 2
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm run dev
```

Open `http://localhost:3000`. The top bar switches robot, controller and grasp
mode. `Physics` is the default SO-101 grasp mode; `Debug Attachment` is visibly
labelled and excluded from physical benchmarks.

## SmolVLA service

The model is not loaded inside FastAPI sessions. Start the independent service
from a dedicated environment:

```powershell
python -m venv .venv-vla
.\.venv-vla\Scripts\pip.exe install -r server/vla_service/requirements.txt
.\.venv-vla\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu130
.\.venv-vla\Scripts\python.exe -m uvicorn vla_service.app:app --app-dir server --port 8100
Invoke-RestMethod -Method Post http://127.0.0.1:8100/load
```

The service reads `VLA_POLICY_PATH` and `VLA_DEVICE` from
`server/.env.local`. The CUDA 13.0 wheel above is the tested Windows setup for
the RTX 2080 SUPER workstation; choose a different official PyTorch wheel when
the target GPU or driver requires it.

The runtime default is `ahmedsohail2003/smolvla-so101-pickplace-v2`, trained on
SO-101 MuJoCo red-block/blue-tray demonstrations with front + wrist RGB. The
single-task checkpoint is conditioned with the exact instruction in
`VLA_TASK_PROMPT`. For best visual-domain matching, fine-tune on the generated
playground dataset. See [SO101_VLA.md](docs/SO101_VLA.md).

## Gemini Robotics ER2

Keep the key only in `server/.env.local`; never use a `NEXT_PUBLIC_*` variable.

```dotenv
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your-key
```

Classical mode exposes only `pick_object` / `place_object`. Hybrid mode exposes
only `execute_vla_subtask({instruction})`; ER2 cannot send joint or Cartesian
commands. Run the standalone access/spatial/function-call test first:

```powershell
server/.venv/Scripts/python.exe scripts/test_gemini_robotics.py
```

## Dataset and fine-tuning

```powershell
# LeRobot Dataset v3, successful physics episodes only (default 200)
python scripts/generate_so101_dataset.py --episodes 200 --root datasets/so101-v3

# Fine-tune; use your local/HF dataset repo id
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=<USER>/so101-v3 --batch_size=8 --steps=20000
```

The generator records both 640×480 cameras, 6D state, 6D applied action and task
at the 30 Hz policy rate. Debug metadata is not part of training observations.

## Tests and benchmarks

```powershell
server/.venv/Scripts/python.exe -m pytest -q
npm run build

server/.venv/Scripts/python.exe scripts/benchmark_so101_classical.py --episodes 100 --grasp-mode physics
server/.venv/Scripts/python.exe scripts/benchmark_so101_vla.py --provider mock
```

The classical benchmark reports grasp, IK, slip/drop and placement failures. It
does not attach or teleport objects. The VLA benchmark refuses to print a task
success rate for `smolvla_base`, mock VLA, or a run without dataset/checkpoint
provenance.

Current development validation:

- 37 passed, 1 opt-in Gemini test skipped
- frontend production build passed
- SO-101: 6 named actuators, 500 Hz MuJoCo, 30 Hz actions, dual RGB cameras
- physical grasp test: both jaw contacts + contact force + object lift, with no attachment
- current 100-scene classical randomized benchmark: **11/100**; below the 90%
  target, honestly retained for controller tuning (62 place/transfer failures,
  15 IK failures and 12 grasp failures)

## Provenance

SO-101 MJCF and STL assets come from
[`TheRobotStudio/SO-ARM100/Simulation/SO101`](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
at commit `7629d2ad9853d10fb903093a33ef6114099d97e5`; the upstream Apache-2.0
license and source record are included beside the model. Panda assets remain as
legacy/reference code.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PROVIDER` | `mock` | Classical high-level planner: `mock` or `gemini` |
| `GEMINI_MODEL` | `gemini-robotics-er-2-preview` | ER2 model ID |
| `VLA_PROVIDER` | `mock` | `mock` integration path or `smolvla` service |
| `VLA_POLICY_PATH` | `ahmedsohail2003/smolvla-so101-pickplace-v2` | SO-101 MuJoCo pick-and-place checkpoint |
| `VLA_TASK_PROMPT` | `Pick up the red block and place it in the blue tray.` | Exact single-task training instruction |
| `VLA_POLICY_JOINT_OFFSETS` | `0,π/2,-π/2,0,π/2,0` | Menagerie-policy to calibrated-simulator joint-zero bridge |
| `VLA_DEVICE` | `cuda` | SmolVLA service device |
| `VLA_HOST` | `http://127.0.0.1:8100` | Internal policy service URL |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Browser-visible API URL |
| `NEXT_PUBLIC_PUBLIC_DEMO` | `false` | Lock the published UI to the safe SO-101 SmolVLA configuration |
| `MAX_AGENT_STEPS` | `15` | Agent/policy action ceiling |
| `VLA_MAX_STEPS` | `240` | SmolVLA rollout cap matching the checkpoint evaluation protocol |
| `MAX_TASK_SECONDS` | `60` | Per-task timeout |
| `MAX_SESSIONS` | `4` | Concurrent simulation sessions |
| `TASKS_PER_HOUR` | `10` | Per-client task rate limit |
| `PUBLIC_VLA_ONLY` | `false` | Restrict an exposed backend to SO-101 + SmolVLA and disable Gemini-backed modes |
