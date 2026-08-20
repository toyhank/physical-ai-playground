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
                qpos, mocap_pos, mocap_quat, finished, result = request
                try:
                    data.qpos[:] = qpos
                    data.qvel[:] = 0.0
                    data.mocap_pos[:] = mocap_pos
                    data.mocap_quat[:] = mocap_quat
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera="robot_camera")
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

    def render(self, qpos: np.ndarray, mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> np.ndarray:
        if not self._ready.wait(timeout=30):
            raise RuntimeError("MUJOCO_RENDERER_START_TIMEOUT")
        if self._startup_error is not None:
            raise RuntimeError("MUJOCO_RENDERER_START_FAILED") from self._startup_error
        if self._closed or not self._thread.is_alive():
            raise RuntimeError("MUJOCO_RENDERER_CLOSED")
        finished = Event()
        result: dict[str, Any] = {}
        self._requests.put((qpos.copy(), mocap_pos.copy(), mocap_quat.copy(), finished, result))
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
        self._cube_qpos_address: int | None = None
        self._cube_dof_address: int | None = None
        self._finger_geom_ids: set[int] = set()
        self._cube_geom_id: int | None = None
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
        if enable_mujoco:
            try:
                import mujoco

                self._mujoco = mujoco
                self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
                self.data = mujoco.MjData(self.model)
                cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free")
                self._cube_qpos_address = int(self.model.jnt_qposadr[cube_joint])
                self._cube_dof_address = int(self.model.jnt_dofadr[cube_joint])
                self._cube_geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom"
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

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        rng = random.Random(seed)
        for _ in range(100):
            cube = np.asarray((rng.uniform(-0.34, -0.08), rng.uniform(0.02, 0.28)), dtype=float)
            box = np.asarray((rng.uniform(0.08, 0.34), rng.uniform(0.02, 0.28)), dtype=float)
            # Keep both objects spatially separated and independently visible
            # from the initial wrist-camera pose. World distance alone can put
            # the container directly in front of the cube in image space.
            if np.linalg.norm(cube - box) >= 0.24 and box[0] - cube[0] >= 0.28:
                break
        self.cube_position = np.asarray((cube[0], cube[1], self.table_z + self.cube_half_size), dtype=float)
        self.box_position = np.asarray((box[0], box[1], self.table_z + 0.012), dtype=float)
        self.ee_position = np.asarray((0.0, -0.10, 0.72), dtype=float)
        self.robot_joints = PANDA_HOME.copy()
        self.gripper_open = True
        self.grasped = False
        self.frame = 0
        self.simulation_steps = 0
        self._sync_mujoco()
        if self.mujoco_enabled and self._gripper_site_id is not None:
            self.ee_position = self.data.site_xpos[self._gripper_site_id].copy()
            self._target_gripper_rotation = self.data.site_xmat[self._gripper_site_id].reshape(3, 3).copy()
        self._latch_observation_camera()
        return self.state()

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
        return {"red_cube": "cube", "blue_box": "box"}.get(body_name)

    def _sync_mujoco(self) -> None:
        if not self.mujoco_enabled:
            return
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.robot_joints
        self.data.qpos[7:9] = 0.04 if self.gripper_open else 0.0
        self.data.ctrl[:7] = self.robot_joints
        self.data.ctrl[7] = 255.0 if self.gripper_open else 0.0
        self._write_cube_pose()
        self.data.mocap_pos[0] = self.box_position
        self._mujoco.mj_forward(self.model, self.data)

    def _write_cube_pose(self) -> None:
        if not self.mujoco_enabled or self._cube_qpos_address is None:
            return
        address = self._cube_qpos_address
        self.data.qpos[address : address + 3] = self.cube_position
        self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        if self._cube_dof_address is not None:
            self.data.qvel[self._cube_dof_address : self._cube_dof_address + 6] = 0.0

    def _step_actuators(self, target_joints: np.ndarray, substeps: int = 10) -> None:
        if not self.mujoco_enabled:
            self.robot_joints = target_joints.copy()
            return
        self.data.ctrl[:7] = target_joints
        self.data.ctrl[7] = 255.0 if self.gripper_open else 0.0
        self.data.mocap_pos[0] = self.box_position
        for _ in range(substeps):
            if self.grasped:
                self._write_cube_pose()
            self._mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
            if self._gripper_site_id is not None:
                self.ee_position = self.data.site_xpos[self._gripper_site_id].copy()
            if self.grasped:
                self.cube_position = self.ee_position.copy()
        self.robot_joints = self.data.qpos[:7].copy()
        if not self.grasped and self._cube_qpos_address is not None:
            address = self._cube_qpos_address
            self.cube_position = self.data.qpos[address : address + 3].copy()

    def _finger_contact_count(self) -> int:
        if not self.mujoco_enabled or self._cube_geom_id is None:
            return 0
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._cube_geom_id in pair and pair.intersection(self._finger_geom_ids):
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
                {"name": "red_cube", "position": [round(float(value), 5) for value in self.cube_position]},
                {"name": "blue_box", "position": [round(float(value), 5) for value in self.box_position]},
            ],
            "grasped": bool(self.grasped),
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
                self.cube_position[:2] if snapped_target == "cube" else self.box_position[:2]
            )
        else:
            for name, object_position in (("cube", self.cube_position), ("box", self.box_position)):
                if np.linalg.norm(target_world[:2] - object_position[:2]) <= 0.08:
                    target_world[:2] = object_position[:2]
                    snapped_target = name
                    break
        if np.any(target_world[:2] < self.workspace_min) or np.any(target_world[:2] > self.workspace_max):
            return ToolExecution(False, "OUTSIDE_ROBOT_WORKSPACE")
        start_position = self.ee_position.copy()
        start_joints = self.robot_joints.copy()
        low_height = (
            self.box_position[2] + 0.012 + self.cube_half_size
            if snapped_target == "box"
            else self.table_z + self.cube_half_size
        )
        end_position = np.asarray((target_world[0], target_world[1], 0.72 if high else low_height), dtype=float)
        if self.mujoco_enabled:
            ik_success, end_joints, ik_error = self._solve_panda_ik(end_position, start_joints)
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
                if self.grasped:
                    self.cube_position = self.ee_position.copy()
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

    def set_gripper_state(self, opened: bool) -> ToolExecution:
        self.gripper_open = opened
        if opened:
            if self.grasped:
                self.grasped = False
                self.cube_position = self.ee_position.copy()
        else:
            distance = float(np.linalg.norm(self.ee_position[:2] - self.cube_position[:2]))
            if distance > 0.065 or self.ee_position[2] > 0.54:
                self.frame += 1
                if self.mujoco_enabled:
                    self._step_actuators(self.robot_joints, substeps=24)
                return ToolExecution(False, "GRASP_FAILED")
        if self.mujoco_enabled:
            if opened and self._cube_qpos_address is not None:
                self._write_cube_pose()
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
            self.cube_position = self.ee_position.copy()
            if self.mujoco_enabled:
                self._write_cube_pose()
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
        delta = np.abs(self.cube_position[:2] - self.box_position[:2])
        xy_inside = bool(np.all(delta <= self.box_half_extents))
        z_inside = self.table_z <= self.cube_position[2] <= self.table_z + 0.12
        return bool(xy_inside and z_inside and not self.grasped)

    def deterministic_pick_place(self) -> list[dict[str, Any]]:
        self._latch_observation_camera()
        cube_x, cube_y = self.world_to_camera_normalized(self.cube_position)
        box_x, box_y = self.world_to_camera_normalized(self.box_position)
        actions = [
            ("set_gripper_state", {"opened": True}),
            ("move", {"x": cube_x, "y": cube_y, "high": True}),
            ("move", {"x": cube_x, "y": cube_y, "high": False}),
            ("set_gripper_state", {"opened": False}),
            ("move", {"x": cube_x, "y": cube_y, "high": True}),
            ("move", {"x": box_x, "y": box_y, "high": True}),
            ("move", {"x": box_x, "y": box_y, "high": False}),
            ("set_gripper_state", {"opened": True}),
            ("move", {"x": box_x, "y": box_y, "high": True}),
        ]
        results = []
        for name, arguments in actions:
            result = getattr(self, name)(**arguments)
            results.append({"name": name, "arguments": arguments, "result": result.as_dict()})
            if not result.success:
                break
        return results

    def camera_png_base64(self, *, latch_observation: bool = True) -> str:
        if latch_observation:
            self._latch_observation_camera()
        if self.mujoco_enabled:
            try:
                pixels = _get_render_service().render(
                    self.data.qpos,
                    self.data.mocap_pos,
                    self.data.mocap_quat,
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
        cube_x, cube_y = self.projector.world_to_normalized(
            self.cube_position, position=position, basis=basis
        )
        box_x, box_y = self.projector.world_to_normalized(
            self.box_position, position=position, basis=basis
        )
        cx, cy = cube_x * 512 // 1000, cube_y * 512 // 1000
        bx, by = box_x * 512 // 1000, box_y * 512 // 1000
        draw.rectangle((cx - 17, cy - 17, cx + 17, cy + 17), fill="#df3029", outline="#8b1511", width=4)
        draw.rectangle((bx - 52, by - 43, bx + 52, by + 43), fill="#8eb2f0", outline="#194eba", width=10)
        return image
