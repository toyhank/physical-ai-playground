# SmolVLA policy service

Use a separate Python 3.12 or 3.13 virtual environment. The service owns the
CUDA model and loads it only once; FastAPI simulation sessions remain small.

```powershell
python -m venv .venv-vla
.\.venv-vla\Scripts\Activate.ps1
pip install -r server/vla_service/requirements.txt
python -m pip install --force-reinstall --no-deps `
  torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu130
uvicorn vla_service.app:app --app-dir server --host 127.0.0.1 --port 8100
```

The service automatically reads `VLA_POLICY_PATH` and `VLA_DEVICE` from the
ignored `server/.env.local` file.

The local default is
`ahmedsohail2003/smolvla-so101-pickplace-v2`, a two-camera SO-101 MuJoCo
pick-and-place checkpoint. `VLA_TASK_PROMPT` intentionally uses the exact
single-task instruction from its training set. `smolvla_base` remains useful
only as a loading/inference smoke test. `VLA_POLICY_JOINT_OFFSETS` converts the
Menagerie SO-ARM100 shoulder/elbow/wrist-roll zero positions to the calibrated
SO-101 MJCF coordinates used by this playground.
