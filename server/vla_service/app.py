from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / "server" / ".env.local", override=False)
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _joint_offsets() -> np.ndarray:
    raw = os.getenv(
        "VLA_POLICY_JOINT_OFFSETS",
        "0,1.5707963268,-1.5707963268,0,1.5707963268,0",
    )
    offsets = np.asarray([float(value.strip()) for value in raw.split(",")])
    if offsets.shape != (6,) or not np.all(np.isfinite(offsets)):
        raise RuntimeError("INVALID_VLA_POLICY_JOINT_OFFSETS")
    return offsets.astype(np.float32)


class InferRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    task: str = Field(min_length=1, max_length=500)
    scene_png_base64: str
    wrist_png_base64: str
    state: list[float] = Field(min_length=6, max_length=6)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class SmolVLARuntime:
    """Load one policy once; never instantiate it inside playground sessions."""

    def __init__(self) -> None:
        self.policy: Any = None
        self.preprocess: Any = None
        self.postprocess: Any = None
        self.device = os.getenv("VLA_DEVICE", "cuda")
        self.policy_path = os.getenv(
            "VLA_POLICY_PATH", "ahmedsohail2003/smolvla-so101-pickplace-v2"
        )
        self.task_prompt = os.getenv(
            "VLA_TASK_PROMPT", "Pick up the red block and place it in the blue tray."
        )
        # The checkpoint uses the MuJoCo Menagerie SO-ARM100 joint zeros.
        # This playground's SO-101 MJCF uses equivalent joints with calibrated
        # shoulder, elbow and wrist-roll zeros. policy + offsets = simulator.
        self.joint_offsets = _joint_offsets()
        self.lock = RLock()

    def load(self) -> dict[str, Any]:
        with self.lock:
            if self.policy is not None:
                return self.status()
            try:
                import torch
                from lerobot.policies import make_pre_post_processors
                from lerobot.policies.smolvla import SmolVLAPolicy
            except ImportError as error:
                raise RuntimeError("LEROBOT_SMOLVLA_NOT_INSTALLED") from error
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA_NOT_AVAILABLE")
            device = torch.device(self.device)
            policy = SmolVLAPolicy.from_pretrained(self.policy_path)
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                self.policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            policy.to(device).eval()
            self.policy, self.preprocess, self.postprocess = policy, preprocess, postprocess
            return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.policy is not None,
            "policy": self.policy_path,
            "device": self.device,
        }

    @staticmethod
    def _image(encoded: str) -> np.ndarray:
        return np.asarray(
            Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        ).copy()

    def infer(self, request: InferRequest) -> list[list[float]]:
        with self.lock:
            if self.policy is None:
                raise RuntimeError("POLICY_NOT_LOADED")
            import torch
            from lerobot.policies.utils import prepare_observation_for_inference

            config = self.policy.config
            image_keys = list(config.image_features)
            if len(image_keys) < 2:
                raise RuntimeError("CHECKPOINT_REQUIRES_FEWER_THAN_TWO_CAMERAS")
            scene = self._image(request.scene_png_base64)
            wrist = self._image(request.wrist_png_base64)
            raw_observation: dict[str, Any] = {
                image_keys[0]: scene,
                image_keys[1]: wrist,
                "observation.state": (
                    np.asarray(request.state, dtype=np.float32) - self.joint_offsets
                ),
            }
            # Match the checkpoint author's evaluation exactly: camera1 is the
            # fixed front view, camera2 is the wrist view, and the unused base
            # camera3 key is omitted rather than turned into a black token.
            frame = prepare_observation_for_inference(
                observation=raw_observation,
                device=torch.device(self.device),
                # This checkpoint is a single-task policy. Keep the user's text
                # in the run log/UI, but condition inference with the exact
                # language used by its training dataset.
                task=self.task_prompt,
                robot_type="so101",
            )
            batch = self.preprocess(frame)
            with torch.inference_mode():
                chunk = self.policy.predict_action_chunk(batch)
            decoded = self.postprocess(chunk)
            array = decoded.detach().cpu().numpy() if hasattr(decoded, "detach") else np.asarray(decoded)
            array = np.asarray(array, dtype=np.float32).reshape(-1, array.shape[-1])
            if array.shape[1] != 6 or not np.all(np.isfinite(array)):
                raise RuntimeError("INVALID_SMOLVLA_ACTION_CHUNK")
            array += self.joint_offsets
            return array.tolist()

    def reset(self) -> None:
        with self.lock:
            if self.policy is not None:
                self.policy.reset()


runtime = SmolVLARuntime()
app = FastAPI(title="Physical AI Playground SmolVLA Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", **runtime.status()}


@app.post("/load")
def load_policy() -> dict[str, Any]:
    try:
        return runtime.load()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/infer")
def infer(request: InferRequest) -> dict[str, Any]:
    try:
        return {
            "action_chunk": runtime.infer(request),
            "policy": runtime.policy_path,
            "representation": "checkpoint_postprocessed",
        }
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/reset")
def reset_session(request: ResetRequest) -> dict[str, Any]:
    del request
    runtime.reset()
    return {"reset": True}
