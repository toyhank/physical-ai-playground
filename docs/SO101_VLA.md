# SO-101 VLA workflow

## 1. What is implemented

The simulator uses the upstream SO-101 MJCF with six position actuators:
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and
`gripper`. State and action tensors are `float32[6]`; MuJoCo uses radians.

The physics loop runs at 500 Hz (`timestep=0.002`). Policy actions run at 30 Hz,
using a repeating 16/17/17 substep schedule. The default grasp mode is contact
physics. The debug attachment mode is never valid for benchmark claims.

## 2. Generate LeRobot Dataset v3

Use a Python 3.12/3.13 environment with `lerobot>=0.6` and video codecs:

```powershell
python -m venv .venv-lerobot
.\.venv-lerobot\Scripts\Activate.ps1
pip install "lerobot[smolvla,dataset]>=0.6,<0.7" mujoco pillow
python scripts/generate_so101_dataset.py --episodes 200 --seed 0 --root datasets/so101-v3 --repo-id <USER>/so101-v3
```

The classical expert may read ground truth and IK, but the saved policy frame is
whitelisted to:

- `observation.images.scene`: 640×480 RGB
- `observation.images.wrist`: 640×480 RGB
- `observation.state`: six joint positions
- `action`: six joint-position targets
- `task`: natural-language instruction

Failed physical episodes are discarded. Start with 200 successful episodes;
increase to 1,000 and 5,000 after inspecting failures and camera coverage.

## 3. Fine-tune SmolVLA

The base checkpoint must be adapted to this camera layout and radian action
schema. A small-memory starting command is:

```powershell
lerobot-train `
  --policy.path=lerobot/smolvla_base `
  --dataset.repo_id=<USER>/so101-v3 `
  --batch_size=8 `
  --steps=20000 `
  --output_dir=outputs/smolvla-so101
```

For an 8 GB GPU, reduce the batch size, use gradient accumulation, mixed
precision and LeRobot's supported PEFT configuration. Full base-model training
is not a realistic 8 GB target. Keep training separate from the playground
runtime.

Do not invent a universal `[-1,1]` or `[-100,100]` conversion. The checkpoint's
pre/post-processors and dataset stats are authoritative. `ActionCodec` supports
identity, min/max and mean/std metadata and validates the final radian limits.

## 4. Start the policy service

```powershell
$env:VLA_POLICY_PATH="outputs/smolvla-so101/checkpoints/last/pretrained_model"
$env:VLA_DEVICE="cuda"
python -m uvicorn vla_service.app:app --app-dir server --host 127.0.0.1 --port 8100
Invoke-RestMethod -Method Post http://127.0.0.1:8100/load
```

Then configure the playground:

```dotenv
VLA_PROVIDER=smolvla
VLA_HOST=http://127.0.0.1:8100
```

The service loads one model and exposes `/health`, `/load`, `/infer`, and
`/reset`. Sessions send only images, state and task. Returned action chunks pass
through the metadata codec, queue and safety filter before MuJoCo.

## 5. Benchmark

First run a chunk-shape smoke test. It does not claim task success:

```powershell
python scripts/benchmark_so101_vla.py --provider smolvla --host http://127.0.0.1:8100 --checkpoint lerobot/smolvla_base
```

Only a fine-tuned run with checkpoint and dataset provenance may report task
success:

```powershell
python scripts/benchmark_so101_vla.py `
  --provider smolvla `
  --checkpoint <USER>/smolvla-so101 `
  --dataset <USER>/so101-v3 `
  --episodes 100 --seed 0 --grasp-mode physics --report-task-success
```

Always record checkpoint, dataset, seed, episode count and grasp mode. Never use
debug attachment to label a run as a physical-grasp benchmark.
