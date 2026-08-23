from __future__ import annotations

import atexit
import base64
import io
import random
from pathlib import Path
from queue import Queue
from threading import Event, RLock, Thread
from typing import Any
from collections.abc import Callable

import numpy as np
from PIL import Image

from app.robot.base import RobotBackend
from app.robot.so101_spec import SO101_JOINTS
from app.simulation.mujoco_engine import CUBE_NAMES, MUJOCO_TO_THREE, ToolExecution


MODEL_PATH = Path(__file__).resolve().parents[2] / "simulation" / "models" / "so101" / "scene.xml"
VLA_REFERENCE_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "simulation"
    / "models"
    / "so101_vla_reference"
    / "pickplace.xml"
)
VLA_POLICY_JOINT_OFFSETS = np.asarray(
    (0.0, np.pi / 2, -np.pi / 2, 0.0, np.pi / 2, 0.0), dtype=np.float32
)
VLA_REFERENCE_JOINT_NAMES = dict(
    zip(
        SO101_JOINTS.names,
        ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"),
        strict=True,
    )
)
SO101_BODY_NAMES = (
    "base",
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
    "moving_jaw_so101_v1",
)


class _SO101RenderService:
    """Keep the Windows OpenGL context on one dedicated thread."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._requests: Queue[Any] = Queue()
        self._ready = Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = Thread(target=self._run, name="so101-renderer", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _run(self) -> None:
        renderer = None
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(self._model_path))
            data = mujoco.MjData(model)
            aligned = self._model_path == VLA_REFERENCE_MODEL_PATH
            renderer = mujoco.Renderer(
                model,
                height=224 if aligned else 480,
                width=224 if aligned else 640,
            )
            self._ready.set()
            while True:
                request = self._requests.get()
                if request is None:
                    break
                qpos, mocap_pos, mocap_quat, camera_name, finished, result = request
                try:
                    data.qpos[:] = qpos
                    data.qvel[:] = 0.0
                    data.mocap_pos[:] = mocap_pos
                    data.mocap_quat[:] = mocap_quat
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera_name)
                    result["pixels"] = renderer.render().copy()
                except BaseException as error:
                    result["error"] = error
                finally:
                    finished.set()
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
        finally:
            if renderer is not None:
                renderer.close()

    def render(
        self,
        qpos: np.ndarray,
        mocap_pos: np.ndarray,
        mocap_quat: np.ndarray,
        camera_name: str,
    ) -> np.ndarray:
        if not self._ready.wait(timeout=30) or self._startup_error:
            raise RuntimeError("SO101_RENDERER_START_FAILED") from self._startup_error
        finished = Event()
        result: dict[str, Any] = {}
        self._requests.put(
            (qpos.copy(), mocap_pos.copy(), mocap_quat.copy(), camera_name, finished, result)
        )
        if not finished.wait(timeout=30):
            raise RuntimeError("SO101_RENDER_TIMEOUT")
        if "error" in result:
            raise RuntimeError("SO101_RENDER_FAILED") from result["error"]
        return result["pixels"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._requests.put(None)
            self._thread.join(timeout=10)


_RENDERERS: dict[Path, _SO101RenderService] = {}
_RENDERER_LOCK = RLock()


def _renderer(model_path: Path) -> _SO101RenderService:
    with _RENDERER_LOCK:
        if model_path not in _RENDERERS:
            _RENDERERS[model_path] = _SO101RenderService(model_path)
        return _RENDERERS[model_path]


class SO101MujocoBackend(RobotBackend):
    robot_id = "so101"
    physics_hz = 500
    policy_hz = 30
    cube_half_size = 0.018
    box_half_extents = np.asarray((0.058, 0.048), dtype=float)

    def __init__(
        self,
        seed: int = 0,
        *,
        grasp_mode: str = "physics",
        vla_aligned: bool = False,
    ) -> None:
        if grasp_mode not in {"physics", "contact_attachment"}:
            raise ValueError("INVALID_GRASP_MODE")
        import mujoco

        self._mujoco = mujoco
        self.vla_aligned = vla_aligned
        self.model_path = VLA_REFERENCE_MODEL_PATH if vla_aligned else MODEL_PATH
        self.policy_hz = 15 if vla_aligned else 30
        self.cube_half_size = 0.012 if vla_aligned else 0.018
        self.box_half_extents = np.asarray(
            (0.037, 0.037) if vla_aligned else (0.058, 0.048), dtype=float
        )
        self._joint_offsets = (
            VLA_POLICY_JOINT_OFFSETS.copy()
            if vla_aligned
            else np.zeros(SO101_JOINTS.size, dtype=np.float32)
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.grasp_mode = grasp_mode
        self._stopped = False
        self._substep_phase = 0
        self.simulation_steps = 0
        self.frame = 0
        self.task_object_id = "red_cube"
        self._active_object_id = "red_cube"
        native_joint_names = (
            VLA_REFERENCE_JOINT_NAMES
            if vla_aligned
            else {name: name for name in SO101_JOINTS.names}
        )
        self._joint_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, native_joint_names[name]
            )
            for name in SO101_JOINTS.names
        }
        self._joint_qpos = {
            name: int(self.model.jnt_qposadr[joint_id])
            for name, joint_id in self._joint_ids.items()
        }
        self._actuator_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, native_joint_names[name]
            )
            for name in SO101_JOINTS.names
        }
        self._cube_joint_addresses: dict[str, int] = {}
        self._cube_dof_addresses: dict[str, int] = {}
        self._cube_geom_ids: dict[str, int] = {}
        self._cube_body_ids: dict[str, int] = {}
        cube_names = ("red_cube",) if vla_aligned else CUBE_NAMES
        for name in cube_names:
            native_name = "block" if vla_aligned else name
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "block_free" if vla_aligned else f"{name}_free",
            )
            self._cube_joint_addresses[name] = int(self.model.jnt_qposadr[joint_id])
            self._cube_dof_addresses[name] = int(self.model.jnt_dofadr[joint_id])
            self._cube_geom_ids[name] = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "block_geom" if vla_aligned else f"{name}_geom",
            )
            self._cube_body_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, native_name
            )
        self._site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "tcp" if vla_aligned else "grasp_site",
        )
        native_body_names = (
            {
                "base": "Base",
                "shoulder": "Rotation_Pitch",
                "upper_arm": "Upper_Arm",
                "lower_arm": "Lower_Arm",
                "wrist": "Wrist_Pitch_Roll",
                "gripper": "Fixed_Jaw",
                "moving_jaw_so101_v1": "Moving_Jaw",
            }
            if vla_aligned
            else {name: name for name in SO101_BODY_NAMES}
        )
        self._body_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, native_body_names[name]
            )
            for name in SO101_BODY_NAMES
        }
        fixed_body = self._body_ids["gripper"]
        moving_body = self._body_ids["moving_jaw_so101_v1"]
        self._fixed_jaw_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == fixed_body
            and int(self.model.geom_contype[geom_id]) != 0
        }
        self._moving_jaw_geoms = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == moving_body
            and int(self.model.geom_contype[geom_id]) != 0
        }
        self._attached_object: str | None = None
        self.frame_observer: Callable[[np.ndarray], None] | None = None
        self._last_metrics: dict[str, Any] = {}
        self.reset(seed)

    @property
    def joint_positions(self) -> np.ndarray:
        native = np.asarray(
            [self.data.qpos[self._joint_qpos[name]] for name in SO101_JOINTS.names],
            dtype=np.float32,
        )
        return native + self._joint_offsets

    def _write_cube_pose(self, name: str, position: np.ndarray) -> None:
        address = self._cube_joint_addresses[name]
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        dof = self._cube_dof_addresses[name]
        self.data.qvel[dof : dof + 6] = 0.0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if self.vla_aligned:
            return self._reset_vla_reference(seed)
        rng = random.Random(seed)
        self._mujoco.mj_resetData(self.model, self.data)
        home = SO101_JOINTS.home_array()
        for index, name in enumerate(SO101_JOINTS.names):
            self.data.qpos[self._joint_qpos[name]] = home[index]
            self.data.ctrl[self._actuator_ids[name]] = home[index]
        box_xy = np.asarray((rng.uniform(0.28, 0.34), rng.uniform(0.10, 0.17)))
        points: list[np.ndarray] = []
        for name in CUBE_NAMES:
            for _ in range(500):
                candidate = np.asarray((rng.uniform(0.17, 0.32), rng.uniform(-0.16, 0.06)))
                if (
                    np.linalg.norm(candidate - box_xy) >= 0.11
                    and all(np.linalg.norm(candidate - other) >= 0.052 for other in points)
                ):
                    points.append(candidate)
                    break
            else:
                raise RuntimeError("SO101_SCENE_RANDOMIZATION_FAILED")
            self._write_cube_pose(
                name, np.asarray((points[-1][0], points[-1][1], self.cube_half_size))
            )
        self.data.mocap_pos[0] = (box_xy[0], box_xy[1], 0.006)
        self.data.mocap_quat[0] = (1.0, 0.0, 0.0, 0.0)
        self._mujoco.mj_forward(self.model, self.data)
        self._stopped = False
        self._attached_object = None
        self.simulation_steps = 0
        self.frame = 0
        self.task_object_id = "red_cube"
        self._active_object_id = "red_cube"
        self._last_metrics = self._physics_metrics()
        return self.get_state()

    def _reset_vla_reference(self, seed: int | None) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        self._mujoco.mj_resetData(self.model, self.data)
        home = self.model.key("home")
        self.data.qpos[:6] = home.qpos[:6]
        self.data.ctrl[:6] = home.ctrl[:6]
        radius = rng.uniform(0.19, 0.27)
        theta = rng.uniform(np.deg2rad(-20), np.deg2rad(35))
        block_xy = np.asarray((radius * np.sin(theta), -radius * np.cos(theta)))
        address = self._cube_joint_addresses["red_cube"]
        self.data.qpos[address : address + 3] = (
            block_xy[0],
            block_xy[1],
            self.cube_half_size,
        )
        half_yaw = 0.5 * np.arctan2(block_xy[0], -block_xy[1])
        self.data.qpos[address + 3 : address + 7] = (
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )
        self._box_position = np.asarray((-0.18, -0.13, 0.006), dtype=float)
        self._mujoco.mj_forward(self.model, self.data)
        self._stopped = False
        self._attached_object = None
        self._substep_phase = 0
        self.simulation_steps = 0
        self.frame = 0
        self.task_object_id = "red_cube"
        self._active_object_id = "red_cube"
        self._last_metrics = self._physics_metrics()
        return self.get_state()

    def _physics_substeps_for_action(self) -> int:
        if self.policy_hz == 15:
            # Match the reference environment exactly:
            # round(1 / (15 Hz * 0.002 s)) == 33 MuJoCo steps per action.
            return 33
        # 500 / 30 = 16.666…; alternate 16, 17, 17 for exact long-run timing.
        pattern = (16, 17, 17)
        value = pattern[self._substep_phase % len(pattern)]
        self._substep_phase += 1
        return value

    def _step(self, target: np.ndarray, *, frames: list[dict[str, Any]] | None = None) -> None:
        self.data.ctrl[:] = np.asarray(target, dtype=float) - self._joint_offsets
        for _ in range(self._physics_substeps_for_action()):
            if self.grasp_mode == "contact_attachment" and self._attached_object:
                position = self.data.site_xpos[self._site_id].copy()
                self._write_cube_pose(self._attached_object, position)
            self._mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
        self.frame += 1
        self._last_metrics = self._physics_metrics()
        if self.frame_observer is not None:
            self.frame_observer(np.asarray(target, dtype=np.float32).copy())
        if frames is not None:
            frames.append(self.get_state())

    def _joint_trajectory(self, target: np.ndarray, steps: int = 18) -> ToolExecution:
        target = np.asarray(target, dtype=float)
        if target.shape != (SO101_JOINTS.size,):
            return ToolExecution(False, "ACTION_SHAPE_MISMATCH")
        if not np.all(np.isfinite(target)):
            return ToolExecution(False, "ACTION_NON_FINITE")
        lower, upper = SO101_JOINTS.lower_array(), SO101_JOINTS.upper_array()
        if np.any(target < lower) or np.any(target > upper):
            return ToolExecution(False, "JOINT_LIMIT")
        start = self.joint_positions.astype(float)
        frames: list[dict[str, Any]] = []
        for fraction in np.linspace(1 / steps, 1.0, steps):
            self._step(start + fraction * (target - start), frames=frames)
        return ToolExecution(True, frames=frames, details={"simulation_steps": self.simulation_steps})

    def _solve_position_ik(
        self, target: np.ndarray, target_rotation: np.ndarray | None = None
    ) -> tuple[bool, np.ndarray, float]:
        work = self._mujoco.MjData(self.model)
        joints = self.joint_positions.astype(float)
        if target_rotation is None:
            target_rotation = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
        lower = SO101_JOINTS.lower_array()[:5].astype(float) + 0.01
        upper = SO101_JOINTS.upper_array()[:5].astype(float) - 0.01
        jac_pos = np.zeros((3, self.model.nv), dtype=float)
        jac_rot = np.zeros((3, self.model.nv), dtype=float)
        orientation_weight = 0.12
        for _ in range(350):
            for index, name in enumerate(SO101_JOINTS.names[:5]):
                work.qpos[self._joint_qpos[name]] = (
                    joints[index] - self._joint_offsets[index]
                )
            self._mujoco.mj_forward(self.model, work)
            error = target - work.site_xpos[self._site_id]
            norm = float(np.linalg.norm(error))
            current_rotation = work.site_xmat[self._site_id].reshape(3, 3)
            rotation_delta = target_rotation @ current_rotation.T
            rotation_error = 0.5 * np.asarray(
                (
                    rotation_delta[2, 1] - rotation_delta[1, 2],
                    rotation_delta[0, 2] - rotation_delta[2, 0],
                    rotation_delta[1, 0] - rotation_delta[0, 1],
                )
            )
            if norm <= 0.003:
                result = self.joint_positions.astype(float)
                result[:5] = joints[:5]
                return True, result, norm
            self._mujoco.mj_jacSite(
                self.model,
                work,
                jac_pos,
                jac_rot,
                self._site_id,
            )
            columns = [int(self.model.jnt_dofadr[self._joint_ids[name]]) for name in SO101_JOINTS.names[:5]]
            jacobian = np.vstack(
                (jac_pos[:, columns], orientation_weight * jac_rot[:, columns])
            )
            combined_error = np.concatenate((error, orientation_weight * rotation_error))
            damped = jacobian @ jacobian.T + 0.0025 * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(damped, combined_error)
            joints[:5] = np.clip(joints[:5] + np.clip(delta, -0.08, 0.08), lower, upper)
        final = float(np.linalg.norm(target - work.site_xpos[self._site_id]))
        result = self.joint_positions.astype(float)
        result[:5] = joints[:5]
        return final <= 0.01, result, final

    def _move_site(self, xy: np.ndarray, high: bool) -> ToolExecution:
        start = self.data.site_xpos[self._site_id].copy()
        target = np.asarray((xy[0], xy[1], 0.105 if high else 0.023), dtype=float)
        frames: list[dict[str, Any]] = []
        last_error = 0.0
        # Solve a Cartesian waypoint path instead of interpolating one distant
        # joint solution. This keeps the under-actuated wrist orientation much
        # steadier while a physically grasped object is being lifted.
        # Limit Cartesian displacement to roughly 6 mm per 30 Hz policy tick.
        # Larger jumps create unrealistic inertial loads that can eject a
        # genuinely contact-grasped object during long lateral transfers.
        segments = max(14, int(np.ceil(np.linalg.norm(target - start) / 0.006)))
        for fraction in np.linspace(1 / segments, 1.0, segments):
            waypoint = start + fraction * (target - start)
            success, joints, last_error = self._solve_position_ik(waypoint)
            if not success:
                return ToolExecution(
                    False,
                    "IK_FAILED",
                    frames=frames,
                    details={"ik_error": round(last_error, 5)},
                )
            step = self._joint_trajectory(joints, steps=1)
            frames.extend(step.frames or [])
            if not step.success:
                return ToolExecution(False, step.error, frames=frames)
        return ToolExecution(
            True,
            frames=frames,
            details={"simulation_steps": self.simulation_steps, "ik_error": round(last_error, 5)},
        )

    def _physics_metrics(self) -> dict[str, Any]:
        cube_geom = self._cube_geom_ids.get(self._active_object_id, -1)
        left_contact = False
        right_contact = False
        normal_force = 0.0
        contacts = 0
        force = np.zeros(6, dtype=float)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if cube_geom not in pair:
                continue
            contacts += 1
            left_contact |= bool(pair.intersection(self._fixed_jaw_geoms))
            right_contact |= bool(pair.intersection(self._moving_jaw_geoms))
            self._mujoco.mj_contactForce(self.model, self.data, index, force)
            normal_force += max(float(force[0]), 0.0)
        dof = self._cube_dof_addresses.get(self._active_object_id)
        relative_speed = 0.0
        if dof is not None:
            site_velocity = np.zeros(6, dtype=float)
            self._mujoco.mj_objectVelocity(
                self.model,
                self.data,
                self._mujoco.mjtObj.mjOBJ_SITE,
                self._site_id,
                site_velocity,
                0,
            )
            relative_speed = float(
                np.linalg.norm(self.data.qvel[dof : dof + 3] - site_velocity[3:])
            )
        height = float(self.data.xpos[self._cube_body_ids[self._active_object_id]][2])
        if left_contact and right_contact and normal_force > 0.1:
            grasp_state = "stable" if relative_speed < 0.15 else "slipping"
        elif left_contact or right_contact:
            grasp_state = "contact"
        elif height > 0.045:
            grasp_state = "lost"
        else:
            grasp_state = "none"
        return {
            "contacts": contacts,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "normal_force": round(normal_force, 5),
            "object_slip_velocity": round(relative_speed, 5),
            "object_height": round(height, 5),
            "grasp_state": grasp_state,
        }

    def _body_states(self) -> list[dict[str, Any]]:
        bodies: list[dict[str, Any]] = []
        for name, body_id in self._body_ids.items():
            rotation = self.data.xmat[body_id].reshape(3, 3)
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = MUJOCO_TO_THREE @ rotation @ MUJOCO_TO_THREE.T
            matrix[:3, 3] = MUJOCO_TO_THREE @ self.data.xpos[body_id]
            bodies.append({"name": name, "matrix": [round(float(v), 7) for v in matrix.reshape(-1)]})
        return bodies

    def get_state(self) -> dict[str, Any]:
        objects = [
            {
                "name": name,
                "position": [round(float(v), 5) for v in self.data.xpos[body_id]],
            }
            for name, body_id in self._cube_body_ids.items()
        ]
        box_position = (
            self._box_position if self.vla_aligned else self.data.mocap_pos[0]
        )
        objects.append(
            {
                "name": "blue_box",
                "position": [round(float(v), 5) for v in box_position],
            }
        )
        return {
            "frame": self.frame,
            "robot_id": self.robot_id,
            "robot": {
                "joints": [round(float(v), 6) for v in self.joint_positions],
                "bodies": self._body_states(),
                "ee_position": [round(float(v), 5) for v in self.data.site_xpos[self._site_id]],
                "gripper_open": bool(self.joint_positions[-1] > 0.7),
            },
            "objects": objects,
            "verified": self.verify_task(),
            "physics": "mujoco",
            "control_mode": "so101-position-actuators",
            "simulation_steps": self.simulation_steps,
            "simulation_time": round(float(self.data.time), 5),
            "actuator_count": int(self.model.nu),
            "policy_fps": self.policy_hz,
            "physics_hz": self.physics_hz,
            "grasp_mode": self.grasp_mode,
            "physics_metrics": self._last_metrics,
            "task_object_id": self.task_object_id,
            "scene_profile": "vla_reference" if self.vla_aligned else "playground",
        }

    def get_observation(self) -> dict[str, Any]:
        return {
            "observation.images.scene": self.render_camera("scene"),
            "observation.images.wrist": self.render_camera("wrist"),
            "observation.state": self.joint_positions.astype(np.float32).tolist(),
        }

    def apply_action(self, action: dict[str, Any]) -> ToolExecution:
        if self._stopped:
            return ToolExecution(False, "SESSION_STOPPED")
        if action.get("name") == "joint_position":
            return self._joint_trajectory(
                np.asarray(action.get("arguments", {}).get("values")), steps=1
            )
        if action.get("name") == "pick_object":
            return self.pick_object(action.get("arguments", {}).get("object_id"))
        if action.get("name") == "place_object":
            return self.place_object(action.get("arguments", {}).get("container_id"))
        return ToolExecution(False, "UNKNOWN_ACTION")

    def _combine(self, skill: str, operations: list[ToolExecution]) -> ToolExecution:
        frames: list[dict[str, Any]] = []
        for operation in operations:
            frames.extend(operation.frames or [])
            if not operation.success:
                return ToolExecution(False, operation.error, frames=frames, details={"skill": skill})
        return ToolExecution(True, frames=frames, details={"skill": skill, "physics_metrics": self._last_metrics})

    def pick_object(self, object_id: str) -> ToolExecution:
        if object_id not in CUBE_NAMES:
            return ToolExecution(False, "UNKNOWN_OBJECT")
        self.task_object_id = object_id
        self._active_object_id = object_id
        xy = self.data.xpos[self._cube_body_ids[object_id]][:2].copy()
        current = self.joint_positions.astype(float)
        opened = current.copy()
        opened[-1] = 1.55
        operations: list[ToolExecution] = []
        for operation in (
            lambda: self._joint_trajectory(opened, steps=8),
            lambda: self._move_site(xy, True),
            lambda: self._move_site(xy, False),
        ):
            result = operation()
            operations.append(result)
            if not result.success:
                return self._combine("pick_object", operations)
        closed = self.joint_positions.astype(float)
        closed[-1] = -0.12
        operations.append(self._joint_trajectory(closed, steps=16))
        if not operations[-1].success:
            return self._combine("pick_object", operations)
        if self.grasp_mode == "contact_attachment" and (
            self._last_metrics["left_contact"] or self._last_metrics["right_contact"]
        ):
            self._attached_object = object_id
        operations.append(self._move_site(xy, True))
        if not operations[-1].success:
            return self._combine("pick_object", operations)
        if self.grasp_mode == "physics" and (
            self._last_metrics["object_height"] < 0.05
            or not self._last_metrics["left_contact"]
            or not self._last_metrics["right_contact"]
        ):
            combined = self._combine("pick_object", operations)
            return ToolExecution(
                False,
                "PHYSICS_GRASP_FAILED",
                frames=combined.frames,
                details={"skill": "pick_object", "physics_metrics": self._last_metrics},
            )
        return self._combine("pick_object", operations)

    def place_object(self, container_id: str) -> ToolExecution:
        if container_id != "blue_box":
            return ToolExecution(False, "UNKNOWN_CONTAINER")
        xy = (
            self._box_position[:2].copy()
            if self.vla_aligned
            else self.data.mocap_pos[0, :2].copy()
        )
        operations: list[ToolExecution] = []
        for operation in (lambda: self._move_site(xy, True), lambda: self._move_site(xy, False)):
            result = operation()
            operations.append(result)
            if not result.success:
                return self._combine("place_object", operations)
        opened = self.joint_positions.astype(float)
        opened[-1] = 1.55
        operations.append(self._joint_trajectory(opened, steps=12))
        self._attached_object = None
        operations.append(self._move_site(xy, True))
        combined = self._combine("place_object", operations)
        if combined.success and not self.verify_task():
            return ToolExecution(
                False,
                "PLACE_VERIFICATION_FAILED",
                frames=combined.frames,
                details={"skill": "place_object", "physics_metrics": self._last_metrics},
            )
        return combined

    def render_camera(self, camera: str, *, latch_observation: bool = False) -> str:
        del latch_observation
        camera_name = (
            {"scene": "front", "wrist": "wrist"}
            if self.vla_aligned
            else {"scene": "scene_camera", "wrist": "wrist_camera"}
        ).get(camera)
        if camera_name is None:
            raise ValueError("UNKNOWN_CAMERA")
        pixels = _renderer(self.model_path).render(
            self.data.qpos, self.data.mocap_pos, self.data.mocap_quat, camera_name
        )
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def verify_task(self) -> bool:
        target = self.data.xpos[self._cube_body_ids[self.task_object_id]]
        box = self._box_position if self.vla_aligned else self.data.mocap_pos[0]
        delta = np.abs(target[:2] - box[:2])
        center_limit = self.box_half_extents - self.cube_half_size - 0.001
        floor_top = float(box[2]) + (0.0 if self.vla_aligned else 0.006)
        expected_height = floor_top + self.cube_half_size
        dof = self._cube_dof_addresses[self.task_object_id]
        linear_speed = float(np.linalg.norm(self.data.qvel[dof : dof + 3]))
        gripper_open = bool(self.joint_positions[-1] > 0.7)
        return bool(
            np.all(delta <= center_limit)
            and abs(float(target[2]) - expected_height) <= 0.008
            and linear_speed <= 0.05
            and gripper_open
            and self._attached_object is None
        )

    def stop(self) -> None:
        self._stopped = True
