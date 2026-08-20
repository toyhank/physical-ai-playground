from __future__ import annotations

import base64
import atexit
import io
import random
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, RLock, Thread
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.robot.ik import PlanarDampedLeastSquaresIK
from app.simulation.camera import FixedCameraProjector


_RENDER_SERVICE_LOCK = RLock()

MODEL_PATH = Path(__file__).resolve().parent / "models" / "franka_panda" / "scene.xml"
PANDA_HOME = np.asarray((0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785), dtype=float)
ROBOT_BODY_NAMES = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
)
CUBE_NAMES = ("red_cube", "green_cube", "yellow_cube", "purple_cube")
CUBE_COLORS = {
    "red_cube": "#df3029",
    "green_cube": "#18a84b",
    "yellow_cube": "#f0ad12",
    "purple_cube": "#8c36c7",
}
MUJOCO_TO_THREE = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))




class _NativeRenderService:
    """Own the Windows OpenGL context on one thread for its full lifetime."""

    def __init__(self) -> None:
        self._requests: Queue[Any] = Queue()
        self._ready = Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = Thread(target=self._run, name="mujoco-renderer", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _run(self) -> None:
        renderer = None
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=512, width=512)
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
                except BaseException as error:  # Propagate native-render failures to the caller.
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
        if not self._ready.wait(timeout=30):
            raise RuntimeError("MUJOCO_RENDERER_START_TIMEOUT")
        if self._startup_error is not None:
            raise RuntimeError("MUJOCO_RENDERER_START_FAILED") from self._startup_error
        if self._closed or not self._thread.is_alive():
            raise RuntimeError("MUJOCO_RENDERER_CLOSED")
        finished = Event()
        result: dict[str, Any] = {}
        self._requests.put(
            (qpos.copy(), mocap_pos.copy(), mocap_quat.copy(), camera_name, finished, result)
        )
        if not finished.wait(timeout=30):
            raise RuntimeError("MUJOCO_RENDER_TIMEOUT")
        if "error" in result:
            raise RuntimeError("MUJOCO_RENDER_FAILED") from result["error"]
        return result["pixels"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._requests.put(None)
            self._thread.join(timeout=10)


_RENDER_SERVICE: _NativeRenderService | None = None


def _get_render_service() -> _NativeRenderService:
    global _RENDER_SERVICE
    with _RENDER_SERVICE_LOCK:
        if _RENDER_SERVICE is None:
            _RENDER_SERVICE = _NativeRenderService()
        return _RENDER_SERVICE


@dataclass
class ToolExecution:
    success: bool
    error: str | None = None
    frames: list[dict[str, Any]] | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.error:
            result["error"] = self.error
        if self.details:
            result.update(self.details)
        return result


class MujocoEngine:
    """MuJoCo-backed scene with a deterministic logical-grasp controller."""

    table_z = 0.44
    cube_half_size = 0.03
    box_half_extents = np.asarray((0.10, 0.08), dtype=float)
    workspace_min = np.asarray((-0.48, -0.05), dtype=float)
    workspace_max = np.asarray((0.48, 0.42), dtype=float)
    robot_base_xy = np.asarray((0.0, -0.37), dtype=float)

    def __init__(self, seed: int = 0, enable_mujoco: bool = True) -> None:
        self.projector = FixedCameraProjector(
            table_z=self.table_z,
            vertical_fov_degrees=86.0,
        )
        self.ik = PlanarDampedLeastSquaresIK()
        self.model = None
        self.data = None
        self._mujoco = None
        self._cube_qpos_addresses: dict[str, int] = {}
        self._cube_dof_addresses: dict[str, int] = {}
        self._finger_geom_ids: set[int] = set()
        self._cube_geom_ids: dict[str, int] = {}
        self._gripper_site_id: int | None = None
        self._camera_id: int | None = None
        self._body_ids: dict[str, int] = {}
        self._target_gripper_rotation: np.ndarray | None = None
        self._observation_camera_pose: tuple[
            np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] | None = None
        self._observation_qpos: np.ndarray | None = None
        self._observation_mocap_pos: np.ndarray | None = None
        self._observation_mocap_quat: np.ndarray | None = None
        self._observed_objects: dict[str, np.ndarray] = {}
        if enable_mujoco:
            try:
                import mujoco

                self._mujoco = mujoco
                self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
                self.data = mujoco.MjData(self.model)
                for cube_name in CUBE_NAMES:
                    cube_joint = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{cube_name}_free"
                    )
                    self._cube_qpos_addresses[cube_name] = int(
                        self.model.jnt_qposadr[cube_joint]
                    )
                    self._cube_dof_addresses[cube_name] = int(
                        self.model.jnt_dofadr[cube_joint]
                    )
                    self._cube_geom_ids[cube_name] = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{cube_name}_geom"
                    )
                finger_body_ids = {
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
                }
                self._finger_geom_ids = {
                    geom_id
                    for geom_id in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[geom_id]) in finger_body_ids
                }
                self._gripper_site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site"
                )
                self._camera_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_camera"
                )
                self._body_ids = {
                    name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    for name in ROBOT_BODY_NAMES
                }
            except (ImportError, OSError, RuntimeError):
                self.model = None
                self.data = None
        self.reset(seed)

    @property
    def mujoco_enabled(self) -> bool:
        return self.model is not None and self.data is not None

    @property
    def cube_position(self) -> np.ndarray:
        """Backward-compatible alias for the red cube used by older tests."""
        return self.cube_positions["red_cube"]

    @cube_position.setter
    def cube_position(self, value: np.ndarray) -> None:
        self.cube_positions["red_cube"] = np.asarray(value, dtype=float)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        rng = random.Random(seed)
        box = np.asarray((rng.uniform(0.21, 0.32), rng.uniform(0.05, 0.26)), dtype=float)
        cube_points: list[np.ndarray] = []
        for _cube_name in CUBE_NAMES:
            for _ in range(500):
                candidate = np.asarray(
                    (rng.uniform(-0.35, 0.03), rng.uniform(0.02, 0.30)), dtype=float
                )
                if (
                    np.linalg.norm(candidate - box) >= 0.20
                    and all(np.linalg.norm(candidate - other) >= 0.105 for other in cube_points)
                ):
                    cube_points.append(candidate)
                    break
            else:
                raise RuntimeError("SCENE_RANDOMIZATION_FAILED")
        self.cube_positions = {
            name: np.asarray((point[0], point[1], self.table_z + self.cube_half_size), dtype=float)
            for name, point in zip(CUBE_NAMES, cube_points, strict=True)
        }
        self.box_position = np.asarray((box[0], box[1], self.table_z + 0.012), dtype=float)
        self.ee_position = np.asarray((0.0, -0.10, 0.72), dtype=float)
        self.robot_joints = PANDA_HOME.copy()
        self.gripper_open = True
        self.grasped = False
        self.held_object_id: str | None = None
        self.task_object_id: str = "red_cube"
        self._active_object_id: str = "red_cube"
        self.frame = 0
        self.simulation_steps = 0
        self._sync_mujoco()
        if self.mujoco_enabled and self._gripper_site_id is not None:
            self.ee_position = self.data.site_xpos[self._gripper_site_id].copy()
            self._target_gripper_rotation = self.data.site_xmat[self._gripper_site_id].reshape(3, 3).copy()
        self._latch_observation_camera()
        self._latch_world_model()
        return self.state()

    def _latch_world_model(self) -> None:
        """Snapshot perception results in the robot base/world frame.

        MuJoCo ground truth stands in for an RGB-D pose estimator in this demo;
        downstream skills consume stable object poses instead of image pixels.
        """
        self._observed_objects = {
            **{name: position.copy() for name, position in self.cube_positions.items()},
            "blue_box": self.box_position.copy(),
        }

    def _current_camera_pose(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.mujoco_enabled and self._camera_id is not None:
            rotation = self.data.cam_xmat[self._camera_id].reshape(3, 3)
            return self.data.cam_xpos[self._camera_id].copy(), (
                rotation[:, 0].copy(),
                rotation[:, 1].copy(),
                -rotation[:, 2].copy(),
            )
        return np.asarray(self.projector.position, dtype=float), self.projector._basis()

    def _latch_observation_camera(self) -> None:
        self._observation_camera_pose = self._current_camera_pose()
        if self.mujoco_enabled:
            self._observation_qpos = self.data.qpos.copy()
            self._observation_mocap_pos = self.data.mocap_pos.copy()
            self._observation_mocap_quat = self.data.mocap_quat.copy()

    def _observation_pose(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._observation_camera_pose is None:
            self._latch_observation_camera()
        assert self._observation_camera_pose is not None
        return self._observation_camera_pose

    def world_to_camera_normalized(
        self, point: np.ndarray | tuple[float, float, float]
    ) -> tuple[int, int]:
        """Project world coordinates into the most recent model observation."""
        position, basis = self._observation_pose()
        return self.projector.world_to_normalized(point, position=position, basis=basis)

    def _observation_ray_target(self, x: int, y: int) -> str | None:
        """Resolve a clicked RGB pixel against the exact MuJoCo observation frame."""
        if (
            not self.mujoco_enabled
            or self._observation_qpos is None
            or self._observation_mocap_pos is None
            or self._observation_mocap_quat is None
        ):
            return None
        camera_position, camera_basis = self._observation_pose()
        origin, ray = self.projector.image_ray(
            x, y, position=camera_position, basis=camera_basis
        )
        observation = self._mujoco.MjData(self.model)
        observation.qpos[:] = self._observation_qpos
        observation.mocap_pos[:] = self._observation_mocap_pos
        observation.mocap_quat[:] = self._observation_mocap_quat
        self._mujoco.mj_forward(self.model, observation)
        geom_id = np.asarray((-1,), dtype=np.int32)
        distance = self._mujoco.mj_ray(
            self.model,
            observation,
            origin,
            ray,
            None,
            True,
            -1,
            geom_id,
        )
        if distance < 0 or geom_id[0] < 0:
            return None
        body_id = int(self.model.geom_bodyid[int(geom_id[0])])
        body_name = self._mujoco.mj_id2name(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return body_name if body_name in (*CUBE_NAMES, "blue_box") else None

    def _sync_mujoco(self) -> None:
        if not self.mujoco_enabled:
            return
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.robot_joints
        self.data.qpos[7:9] = 0.04 if self.gripper_open else 0.0
        self.data.ctrl[:7] = self.robot_joints
        self.data.ctrl[7] = 255.0 if self.gripper_open else 0.0
        for cube_name in CUBE_NAMES:
            self._write_cube_pose(cube_name)
        self.data.mocap_pos[0] = self.box_position
        self._mujoco.mj_forward(self.model, self.data)

    def _write_cube_pose(self, cube_name: str) -> None:
        address = self._cube_qpos_addresses.get(cube_name)
        if not self.mujoco_enabled or address is None:
            return
        self.data.qpos[address : address + 3] = self.cube_positions[cube_name]
        self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        dof_address = self._cube_dof_addresses.get(cube_name)
        if dof_address is not None:
            self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _step_actuators(self, target_joints: np.ndarray, substeps: int = 10) -> None:
        if not self.mujoco_enabled:
            self.robot_joints = target_joints.copy()
            return
        self.data.ctrl[:7] = target_joints
        self.data.ctrl[7] = 255.0 if self.gripper_open else 0.0
        self.data.mocap_pos[0] = self.box_position
        for _ in range(substeps):
            if self.grasped and self.held_object_id:
                self._write_cube_pose(self.held_object_id)
            self._mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
            if self._gripper_site_id is not None:
                self.ee_position = self.data.site_xpos[self._gripper_site_id].copy()
            if self.grasped and self.held_object_id:
                self.cube_positions[self.held_object_id] = self.ee_position.copy()
        self.robot_joints = self.data.qpos[:7].copy()
        for cube_name, address in self._cube_qpos_addresses.items():
            if not self.grasped or cube_name != self.held_object_id:
                self.cube_positions[cube_name] = self.data.qpos[address : address + 3].copy()

    def _finger_contact_count(self) -> int:
        cube_geom_id = self._cube_geom_ids.get(self._active_object_id)
        if not self.mujoco_enabled or cube_geom_id is None:
            return 0
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if cube_geom_id in pair and pair.intersection(self._finger_geom_ids):
                count += 1
        return count

    def _robot_body_states(self) -> list[dict[str, Any]]:
        if not self.mujoco_enabled:
            return []
        bodies: list[dict[str, Any]] = []
        for name, body_id in self._body_ids.items():
            rotation = self.data.xmat[body_id].reshape(3, 3)
            three_rotation = MUJOCO_TO_THREE @ rotation @ MUJOCO_TO_THREE.T
            three_position = MUJOCO_TO_THREE @ self.data.xpos[body_id]
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = three_rotation
            matrix[:3, 3] = three_position
            bodies.append(
                {
                    "name": name,
                    "matrix": [round(float(value), 7) for value in matrix.reshape(-1)],
                }
            )
        return bodies

    @staticmethod
    def _orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        delta = target @ current.T
        return 0.5 * np.asarray(
            (delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]),
            dtype=float,
        )

    def _solve_panda_ik(self, target: np.ndarray, initial: np.ndarray) -> tuple[bool, np.ndarray, float]:
        if not self.mujoco_enabled or self._gripper_site_id is None or self._target_gripper_rotation is None:
            return False, initial.copy(), float("inf")
        work = self._mujoco.MjData(self.model)
        joints = initial.copy()
        lower = self.model.jnt_range[:7, 0] + 0.015
        upper = self.model.jnt_range[:7, 1] - 0.015
        jacobian_position = np.zeros((3, self.model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=float)
        orientation_weight = 0.32
        damping = 0.045
        for _ in range(260):
            work.qpos[:7] = joints
            work.qpos[7:9] = 0.04
            self._mujoco.mj_forward(self.model, work)
            position_error = target - work.site_xpos[self._gripper_site_id]
            current_rotation = work.site_xmat[self._gripper_site_id].reshape(3, 3)
            rotation_error = self._orientation_error(self._target_gripper_rotation, current_rotation)
            position_norm = float(np.linalg.norm(position_error))
            if position_norm <= 0.0035 and np.linalg.norm(rotation_error) <= 0.055:
                return True, joints, position_norm
            self._mujoco.mj_jacSite(
                self.model,
                work,
                jacobian_position,
                jacobian_rotation,
                self._gripper_site_id,
            )
            jacobian = np.vstack(
                (jacobian_position[:, :7], orientation_weight * jacobian_rotation[:, :7])
            )
            error = np.concatenate((position_error, orientation_weight * rotation_error))
            damped = jacobian @ jacobian.T + (damping**2) * np.eye(6)
            pseudo_inverse = jacobian.T @ np.linalg.solve(damped, np.eye(6))
            delta = pseudo_inverse @ error
            nullspace = np.eye(7) - pseudo_inverse @ jacobian
            delta += nullspace @ (0.035 * (PANDA_HOME - joints))
            joints = np.clip(joints + np.clip(delta, -0.09, 0.09), lower, upper)
        work.qpos[:7] = joints
        self._mujoco.mj_forward(self.model, work)
        final_error = float(np.linalg.norm(target - work.site_xpos[self._gripper_site_id]))
        return final_error <= 0.012, joints, final_error

    def state(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "robot": {
                "joints": [round(float(value), 5) for value in self.robot_joints],
                "finger_joints": (
                    [round(float(value), 5) for value in self.data.qpos[7:9]]
                    if self.mujoco_enabled
                    else [0.04 if self.gripper_open else 0.0] * 2
                ),
                "bodies": self._robot_body_states(),
                "ee_position": [round(float(value), 5) for value in self.ee_position],
                "gripper_open": bool(self.gripper_open),
            },
            "objects": [
                *[
                    {
                        "name": name,
                        "position": [round(float(value), 5) for value in self.cube_positions[name]],
                    }
                    for name in CUBE_NAMES
                ],
                {"name": "blue_box", "position": [round(float(value), 5) for value in self.box_position]},
            ],
            "grasped": bool(self.grasped),
            "held_object_id": self.held_object_id,
            "task_object_id": self.task_object_id,
            "verified": self.verify_task(),
            "physics": "mujoco" if self.mujoco_enabled else "kinematic-fallback",
            "control_mode": "actuator-mj_step" if self.mujoco_enabled else "direct-kinematics",
            "simulation_steps": self.simulation_steps,
            "simulation_time": round(float(self.data.time), 4) if self.mujoco_enabled else 0.0,
            "actuator_count": int(self.model.nu) if self.mujoco_enabled else 0,
        }

    def move(self, x: int, y: int, high: bool) -> ToolExecution:
        camera_position, camera_basis = self._observation_pose()
        target_world = self.projector.image_point_to_world(
            x,
            y,
            position=camera_position,
            basis=camera_basis,
        )
        # Robotics ER reports approximate normalized image points. When that
        # projection lands close to a known scene object, lock the Cartesian
        # target to its measured MuJoCo pose so the last centimetres of the
        # approach are handled by physics instead of projection error.
        snapped_target = self._observation_ray_target(x, y)
        if snapped_target is not None:
            target_world[:2] = (
                self.box_position[:2]
                if snapped_target == "blue_box"
                else self.cube_positions[snapped_target][:2]
            )
        else:
            scene_objects = [
                *((name, self.cube_positions[name]) for name in CUBE_NAMES),
                ("blue_box", self.box_position),
            ]
            for name, object_position in scene_objects:
                if np.linalg.norm(target_world[:2] - object_position[:2]) <= 0.08:
                    target_world[:2] = object_position[:2]
                    snapped_target = name
                    break
        if snapped_target in CUBE_NAMES:
            self._active_object_id = snapped_target
        return self._move_to_world(target_world[:2], high, snapped_target)

    def _move_to_world(
        self,
        target_xy: np.ndarray | tuple[float, float] | list[float],
        high: bool,
        snapped_target: str | None = None,
    ) -> ToolExecution:
        target_world = np.asarray(target_xy, dtype=float)
        if np.any(target_world < self.workspace_min) or np.any(target_world > self.workspace_max):
            return ToolExecution(False, "OUTSIDE_ROBOT_WORKSPACE")
        start_position = self.ee_position.copy()
        start_joints = self.robot_joints.copy()
        low_height = (
            self.box_position[2] + 0.012 + self.cube_half_size
            if snapped_target == "blue_box"
            else self.table_z + self.cube_half_size
        )
        end_position = np.asarray((target_world[0], target_world[1], 0.72 if high else low_height), dtype=float)
        if self.mujoco_enabled:
            ik_success, end_joints, ik_error = self._solve_panda_ik(end_position, start_joints)
            if not ik_success:
                # A carried object can leave the arm in a poor local IK basin.
                # Re-seed from Panda's standard collision-clear posture while
                # retaining the same Cartesian goal and tool orientation.
                retry_success, retry_joints, retry_error = self._solve_panda_ik(
                    end_position, PANDA_HOME
                )
                if retry_success or retry_error < ik_error:
                    ik_success, end_joints, ik_error = (
                        retry_success,
                        retry_joints,
                        retry_error,
                    )
            if not ik_success:
                return ToolExecution(False, "IK_FAILED", details={"ik_error": round(ik_error, 5)})
        else:
            planar_target = target_world[:2] - self.robot_base_xy
            ik_result = self.ik.solve(planar_target, self.robot_joints[:3])
            if not ik_result.success:
                return ToolExecution(False, "IK_FAILED")
            end_joints = start_joints.copy()
            end_joints[:3] = ik_result.joints
            ik_error = ik_result.error
        frames: list[dict[str, Any]] = []
        for fraction in np.linspace(1.0 / 12.0, 1.0, 12):
            requested_position = start_position + fraction * (end_position - start_position)
            target_joints = start_joints + fraction * (end_joints - start_joints)
            self.frame += 1
            if self.mujoco_enabled:
                self._step_actuators(target_joints, substeps=20)
            else:
                self.robot_joints = target_joints
                self.ee_position = requested_position
                if self.grasped and self.held_object_id:
                    self.cube_positions[self.held_object_id] = self.ee_position.copy()
            frames.append(self.state())
        return ToolExecution(
            True,
            frames=frames,
            details={
                "control_mode": "actuator-mj_step" if self.mujoco_enabled else "direct-kinematics",
                "ik_error": round(float(ik_error), 5),
                "simulation_steps": self.simulation_steps,
            },
        )

    def _execute_skill(
        self, skill: str, actions: list[tuple[str, dict[str, Any]]]
    ) -> ToolExecution:
        frames: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        for name, arguments in actions:
            if name == "move_world":
                result = self._move_to_world(**arguments)
            else:
                result = self.set_gripper_state(**arguments)
            frames.extend(result.frames or [])
            stages.append({"name": name, "arguments": arguments, "result": result.as_dict()})
            if not result.success:
                return ToolExecution(
                    False,
                    result.error,
                    frames=frames,
                    details={"skill": skill, "failed_stage": name, "stages": stages},
                )
        return ToolExecution(
            True,
            frames=frames,
            details={"skill": skill, "stages": stages, "simulation_steps": self.simulation_steps},
        )

    def pick_object(self, object_id: str) -> ToolExecution:
        if object_id not in CUBE_NAMES:
            return ToolExecution(False, "UNKNOWN_OBJECT")
        if self.grasped:
            return ToolExecution(False, "GRIPPER_ALREADY_HOLDING_OBJECT")
        target = self._observed_objects.get(object_id)
        if target is None:
            return ToolExecution(False, "OBJECT_NOT_OBSERVED")
        self.task_object_id = object_id
        self._active_object_id = object_id
        xy = [float(target[0]), float(target[1])]
        return self._execute_skill(
            "pick_object",
            [
                ("gripper", {"opened": True}),
                ("move_world", {"target_xy": xy, "high": True, "snapped_target": object_id}),
                ("move_world", {"target_xy": xy, "high": False, "snapped_target": object_id}),
                ("gripper", {"opened": False}),
                ("move_world", {"target_xy": xy, "high": True, "snapped_target": object_id}),
            ],
        )

    def place_object(self, container_id: str) -> ToolExecution:
        if container_id != "blue_box":
            return ToolExecution(False, "UNKNOWN_CONTAINER")
        if not self.grasped:
            return ToolExecution(False, "NOT_HOLDING_OBJECT")
        target = self._observed_objects.get(container_id)
        if target is None:
            return ToolExecution(False, "CONTAINER_NOT_OBSERVED")
        xy = [float(target[0]), float(target[1])]
        return self._execute_skill(
            "place_object",
            [
                ("move_world", {"target_xy": xy, "high": True, "snapped_target": "blue_box"}),
                ("move_world", {"target_xy": xy, "high": False, "snapped_target": "blue_box"}),
                ("gripper", {"opened": True}),
                ("move_world", {"target_xy": xy, "high": True, "snapped_target": "blue_box"}),
            ],
        )

    def set_gripper_state(self, opened: bool) -> ToolExecution:
        self.gripper_open = opened
        released_object_id: str | None = None
        if opened:
            if self.grasped and self.held_object_id:
                released_object_id = self.held_object_id
                self.grasped = False
                self.cube_positions[released_object_id] = self.ee_position.copy()
                self.held_object_id = None
        else:
            active_position = self.cube_positions[self._active_object_id]
            distance = float(np.linalg.norm(self.ee_position[:2] - active_position[:2]))
            if distance > 0.065 or self.ee_position[2] > 0.54:
                self.frame += 1
                if self.mujoco_enabled:
                    self._step_actuators(self.robot_joints, substeps=24)
                return ToolExecution(False, "GRASP_FAILED")
        if self.mujoco_enabled:
            if opened and released_object_id:
                self._write_cube_pose(released_object_id)
                self._mujoco.mj_forward(self.model, self.data)
            if opened:
                self._step_actuators(self.robot_joints, substeps=36)
            else:
                # Contact can be brief before a small free cube is squeezed
                # away. Detect it at simulation-step resolution and latch the
                # stabilized grasp at the first genuine finger contact.
                contact_count = 0
                for _ in range(200):
                    self._step_actuators(self.robot_joints, substeps=1)
                    contact_count = max(contact_count, self._finger_contact_count())
                    if contact_count:
                        break
        else:
            contact_count = 0
        if opened:
            contact_count = self._finger_contact_count()
        if not opened:
            if self.mujoco_enabled and contact_count == 0:
                self.frame += 1
                return ToolExecution(
                    False,
                    "NO_FINGER_CONTACT",
                    frames=[self.state()],
                    details={
                        "contact_count": 0,
                        "finger_actuated": True,
                        "simulation_steps": self.simulation_steps,
                    },
                )
            self.grasped = True
            self.held_object_id = self._active_object_id
            self.cube_positions[self.held_object_id] = self.ee_position.copy()
            if self.mujoco_enabled:
                self._write_cube_pose(self.held_object_id)
                self._mujoco.mj_forward(self.model, self.data)
        self.frame += 1
        return ToolExecution(
            True,
            frames=[self.state()],
            details={
                "contact_count": contact_count,
                "finger_actuated": self.mujoco_enabled,
                "grasp_constraint": "contact-gated-attachment" if not opened else "released",
                "simulation_steps": self.simulation_steps,
            },
        )

    def verify_task(self) -> bool:
        target_position = self.cube_positions[self.task_object_id]
        delta = np.abs(target_position[:2] - self.box_position[:2])
        xy_inside = bool(np.all(delta <= self.box_half_extents))
        z_inside = self.table_z <= target_position[2] <= self.table_z + 0.12
        return bool(xy_inside and z_inside and not self.grasped)

    def deterministic_pick_place(self) -> list[dict[str, Any]]:
        self._latch_world_model()
        actions = [
            ("pick_object", {"object_id": "red_cube"}),
            ("place_object", {"container_id": "blue_box"}),
        ]
        results = []
        for name, arguments in actions:
            result = getattr(self, name)(**arguments)
            results.append({"name": name, "arguments": arguments, "result": result.as_dict()})
            if not result.success:
                break
        return results

    def camera_png_base64(
        self, *, camera: str = "wrist", latch_observation: bool = True
    ) -> str:
        camera_name = {"wrist": "robot_camera", "scene": "scene_camera"}.get(camera)
        if camera_name is None:
            raise ValueError("UNKNOWN_CAMERA")
        if latch_observation:
            self._latch_world_model()
            if camera == "wrist":
                self._latch_observation_camera()
        if self.mujoco_enabled:
            try:
                pixels = _get_render_service().render(
                    self.data.qpos,
                    self.data.mocap_pos,
                    self.data.mocap_quat,
                    camera_name,
                )
                image = Image.fromarray(pixels)
                return self._encode_png(image)
            except (RuntimeError, OSError):
                pass
        return self._encode_png(self._fallback_camera_image(live=not latch_observation))

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _fallback_camera_image(self, *, live: bool = False) -> Image.Image:
        image = Image.new("RGB", (512, 512), "#c9d2c9")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((28, 28, 484, 484), radius=18, fill="#d6c7ad", outline="#7c776b", width=4)
        if live:
            position, basis = self._current_camera_pose()
        else:
            position, basis = self._observation_pose()
        box_x, box_y = self.projector.world_to_normalized(
            self.box_position, position=position, basis=basis
        )
        bx, by = box_x * 512 // 1000, box_y * 512 // 1000
        for cube_name, cube_position in self.cube_positions.items():
            cube_x, cube_y = self.projector.world_to_normalized(
                cube_position, position=position, basis=basis
            )
            cx, cy = cube_x * 512 // 1000, cube_y * 512 // 1000
            draw.rectangle(
                (cx - 17, cy - 17, cx + 17, cy + 17),
                fill=CUBE_COLORS[cube_name],
                outline="#342f2d",
                width=3,
            )
        draw.rectangle((bx - 52, by - 43, bx + 52, by + 43), fill="#8eb2f0", outline="#194eba", width=10)
        return image
