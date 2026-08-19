from __future__ import annotations

import base64
import atexit
import io
import random
from dataclasses import dataclass
from queue import Queue
from threading import Event, RLock, Thread
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.robot.ik import PlanarDampedLeastSquaresIK
from app.simulation.camera import FixedCameraProjector


_RENDER_SERVICE_LOCK = RLock()


MJCF = r"""
<mujoco model="physical_ai_playground">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual><global offwidth="512" offheight="512"/><quality shadowsize="2048"/></visual>
  <default>
    <joint damping="2" armature="0.05" limited="true"/>
    <geom friction="1 0.1 0.01" condim="3"/>
  </default>
  <asset>
    <material name="table" rgba="0.72 0.62 0.48 1"/>
    <material name="robot" rgba="0.82 0.86 0.84 1"/>
    <material name="dark" rgba="0.12 0.17 0.15 1"/>
    <material name="red" rgba="0.88 0.10 0.08 1"/>
    <material name="blue" rgba="0.06 0.26 0.88 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.4 2.4" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.80 0.84 0.80 1"/>
    <body name="table" pos="0 0 0.40">
      <geom type="box" size="0.58 0.46 0.04" material="table"/>
    </body>
    <body name="panda_link0" pos="0 -0.37 0.44">
      <geom type="cylinder" size="0.10 0.07" material="dark"/>
      <body name="panda_link1" pos="0 0 0.07">
        <joint name="panda_joint1" axis="0 0 1" range="-2.90 2.90"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.055" material="robot"/>
        <body name="panda_link2" pos="0 0 0.18">
          <joint name="panda_joint2" axis="0 1 0" range="-2.10 2.10"/>
          <geom type="capsule" fromto="0 0 0 0.20 0 0.05" size="0.05" material="robot"/>
          <body name="panda_link3" pos="0.20 0 0.05">
            <joint name="panda_joint3" axis="0 0 1" range="-2.90 2.90"/>
            <geom type="capsule" fromto="0 0 0 0.19 0 0" size="0.047" material="robot"/>
            <body name="panda_link4" pos="0.19 0 0">
              <joint name="panda_joint4" axis="0 1 0" range="-3.00 0.10"/>
              <geom type="capsule" fromto="0 0 0 0.14 0 -0.05" size="0.043" material="robot"/>
              <body name="panda_link5" pos="0.14 0 -0.05">
                <joint name="panda_joint5" axis="0 0 1" range="-2.90 2.90"/>
                <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.04" material="robot"/>
                <body name="panda_link6" pos="0.10 0 0">
                  <joint name="panda_joint6" axis="0 1 0" range="-0.10 3.70"/>
                  <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.037" material="robot"/>
                  <body name="panda_link7" pos="0.08 0 0">
                    <joint name="panda_joint7" axis="0 0 1" range="-2.90 2.90"/>
                    <geom type="cylinder" size="0.05 0.04" material="dark"/>
                    <body name="panda_hand" pos="0.07 0 0">
                      <geom type="cylinder" size="0.055 0.045" material="dark"/>
                      <body name="left_finger">
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04" damping="1"/>
                        <geom name="left_finger_geom" type="box" pos="0.045 0.04 -0.035" size="0.045 0.01 0.055" material="dark" friction="2 0.1 0.01"/>
                        <site name="left_finger_touch" pos="0.06 0.028 -0.035" size="0.018"/>
                      </body>
                      <body name="right_finger">
                        <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04" damping="1"/>
                        <geom name="right_finger_geom" type="box" pos="0.045 -0.04 -0.035" size="0.045 0.01 0.055" material="dark" friction="2 0.1 0.01"/>
                        <site name="right_finger_touch" pos="0.06 -0.028 -0.035" size="0.018"/>
                      </body>
                      <site name="gripper_site" pos="0.09 0 -0.035" size="0.018" rgba="0.8 1 0.3 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="red_cube" pos="-0.20 0.10 0.47">
      <freejoint name="red_cube_free"/>
      <geom name="red_cube_geom" type="box" size="0.03 0.03 0.03" material="red" mass="0.08"/>
    </body>
    <body name="blue_box" mocap="true" pos="0.22 0.12 0.46">
      <geom type="box" pos="0 0 0" size="0.12 0.10 0.012" material="blue"/>
      <geom type="box" pos="0.11 0 0.055" size="0.01 0.10 0.055" material="blue"/>
      <geom type="box" pos="-0.11 0 0.055" size="0.01 0.10 0.055" material="blue"/>
      <geom type="box" pos="0 0.09 0.055" size="0.10 0.01 0.055" material="blue"/>
      <geom type="box" pos="0 -0.09 0.055" size="0.10 0.01 0.055" material="blue"/>
    </body>
    <camera name="robot_camera" pos="0 -1.15 1.35" xyaxes="1 0 0 0 0.605 0.796" fovy="49"/>
  </worldbody>
  <actuator>
    <position name="joint1_motor" joint="panda_joint1" kp="120" ctrlrange="-2.90 2.90" forcerange="-90 90"/>
    <position name="joint2_motor" joint="panda_joint2" kp="120" ctrlrange="-2.10 2.10" forcerange="-90 90"/>
    <position name="joint3_motor" joint="panda_joint3" kp="100" ctrlrange="-2.90 2.90" forcerange="-70 70"/>
    <position name="joint4_motor" joint="panda_joint4" kp="90" ctrlrange="-3.00 0.10" forcerange="-55 55"/>
    <position name="joint5_motor" joint="panda_joint5" kp="70" ctrlrange="-2.90 2.90" forcerange="-40 40"/>
    <position name="joint6_motor" joint="panda_joint6" kp="60" ctrlrange="-0.10 3.70" forcerange="-35 35"/>
    <position name="joint7_motor" joint="panda_joint7" kp="45" ctrlrange="-2.90 2.90" forcerange="-25 25"/>
    <position name="left_finger_motor" joint="finger_joint1" kp="180" ctrlrange="0 0.04" forcerange="-20 20"/>
    <position name="right_finger_motor" joint="finger_joint2" kp="180" ctrlrange="0 0.04" forcerange="-20 20"/>
  </actuator>
  <sensor>
    <touch name="left_finger_contact" site="left_finger_touch"/>
    <touch name="right_finger_contact" site="right_finger_touch"/>
  </sensor>
</mujoco>
"""


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

            model = mujoco.MjModel.from_xml_string(MJCF)
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
        self.projector = FixedCameraProjector(table_z=self.table_z)
        self.ik = PlanarDampedLeastSquaresIK()
        self.model = None
        self.data = None
        self._mujoco = None
        self._cube_qpos_address: int | None = None
        self._cube_dof_address: int | None = None
        self._finger_geom_ids: set[int] = set()
        self._cube_geom_id: int | None = None
        if enable_mujoco:
            try:
                import mujoco

                self._mujoco = mujoco
                self.model = mujoco.MjModel.from_xml_string(MJCF)
                self.data = mujoco.MjData(self.model)
                cube_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free")
                self._cube_qpos_address = int(self.model.jnt_qposadr[cube_joint])
                self._cube_dof_address = int(self.model.jnt_dofadr[cube_joint])
                self._cube_geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube_geom"
                )
                self._finger_geom_ids = {
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"),
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"),
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
            if np.linalg.norm(cube - box) >= 0.24:
                break
        self.cube_position = np.asarray((cube[0], cube[1], self.table_z + self.cube_half_size), dtype=float)
        self.box_position = np.asarray((box[0], box[1], self.table_z + 0.012), dtype=float)
        self.ee_position = np.asarray((0.0, -0.10, 0.72), dtype=float)
        self.robot_joints = np.asarray((0.3, 0.8, -1.0, -1.5, 0.0, 1.8, 0.7), dtype=float)
        self.gripper_open = True
        self.grasped = False
        self.frame = 0
        self.simulation_steps = 0
        self._sync_mujoco()
        return self.state()

    def _sync_mujoco(self) -> None:
        if not self.mujoco_enabled:
            return
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.robot_joints
        self.data.qpos[7:9] = 0.04 if self.gripper_open else 0.0
        self.data.ctrl[:7] = self.robot_joints
        self.data.ctrl[7:9] = 0.04 if self.gripper_open else 0.0
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
        self.data.ctrl[7:9] = 0.04 if self.gripper_open else 0.0
        self.data.mocap_pos[0] = self.box_position
        for _ in range(substeps):
            if self.grasped:
                self._write_cube_pose()
            self._mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
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

    def state(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "robot": {
                "joints": [round(float(value), 5) for value in self.robot_joints],
                "ee_position": [round(float(value), 5) for value in self.ee_position],
                "gripper_open": self.gripper_open,
            },
            "objects": [
                {"name": "red_cube", "position": [round(float(value), 5) for value in self.cube_position]},
                {"name": "blue_box", "position": [round(float(value), 5) for value in self.box_position]},
            ],
            "grasped": self.grasped,
            "verified": self.verify_task(),
            "physics": "mujoco" if self.mujoco_enabled else "kinematic-fallback",
            "control_mode": "actuator-mj_step" if self.mujoco_enabled else "direct-kinematics",
            "simulation_steps": self.simulation_steps,
            "simulation_time": round(float(self.data.time), 4) if self.mujoco_enabled else 0.0,
            "actuator_count": int(self.model.nu) if self.mujoco_enabled else 0,
        }

    def move(self, x: int, y: int, high: bool) -> ToolExecution:
        target_world = self.projector.image_point_to_world(x, y)
        if np.any(target_world[:2] < self.workspace_min) or np.any(target_world[:2] > self.workspace_max):
            return ToolExecution(False, "OUTSIDE_ROBOT_WORKSPACE")
        planar_target = target_world[:2] - self.robot_base_xy
        ik_result = self.ik.solve(planar_target, self.robot_joints[:3])
        if not ik_result.success:
            return ToolExecution(False, "IK_FAILED")
        start_position = self.ee_position.copy()
        start_joints = self.robot_joints.copy()
        end_position = np.asarray((target_world[0], target_world[1], 0.70 if high else 0.505), dtype=float)
        end_joints = start_joints.copy()
        end_joints[:3] = ik_result.joints
        frames: list[dict[str, Any]] = []
        for fraction in np.linspace(0.1, 1.0, 10):
            self.ee_position = start_position + fraction * (end_position - start_position)
            target_joints = start_joints + fraction * (end_joints - start_joints)
            if self.grasped:
                self.cube_position = self.ee_position + np.asarray((0.0, 0.0, -0.04))
            self.frame += 1
            if self.mujoco_enabled:
                self._step_actuators(target_joints, substeps=12)
            else:
                self.robot_joints = target_joints
            frames.append(self.state())
        return ToolExecution(
            True,
            frames=frames,
            details={
                "control_mode": "actuator-mj_step" if self.mujoco_enabled else "direct-kinematics",
                "simulation_steps": self.simulation_steps,
            },
        )

    def set_gripper_state(self, opened: bool) -> ToolExecution:
        self.gripper_open = opened
        if opened:
            if self.grasped:
                self.grasped = False
                self.cube_position = np.asarray(
                    (self.ee_position[0], self.ee_position[1], self.table_z + self.cube_half_size), dtype=float
                )
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
            self._step_actuators(self.robot_joints, substeps=36)
        contact_count = self._finger_contact_count()
        if not opened:
            self.grasped = True
            self.cube_position = self.ee_position + np.asarray((0.0, 0.0, -0.04))
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
                "simulation_steps": self.simulation_steps,
            },
        )

    def verify_task(self) -> bool:
        delta = np.abs(self.cube_position[:2] - self.box_position[:2])
        xy_inside = bool(np.all(delta <= self.box_half_extents))
        z_inside = self.table_z <= self.cube_position[2] <= self.table_z + 0.12
        return xy_inside and z_inside and not self.grasped

    def deterministic_pick_place(self) -> list[dict[str, Any]]:
        cube_x, cube_y = self.projector.world_to_normalized(self.cube_position)
        box_x, box_y = self.projector.world_to_normalized(self.box_position)
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

    def camera_png_base64(self) -> str:
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
        return self._encode_png(self._fallback_camera_image())

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _fallback_camera_image(self) -> Image.Image:
        image = Image.new("RGB", (512, 512), "#c9d2c9")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((28, 28, 484, 484), radius=18, fill="#d6c7ad", outline="#7c776b", width=4)
        cube_x, cube_y = self.projector.world_to_normalized(self.cube_position)
        box_x, box_y = self.projector.world_to_normalized(self.box_position)
        cx, cy = cube_x * 512 // 1000, cube_y * 512 // 1000
        bx, by = box_x * 512 // 1000, box_y * 512 // 1000
        draw.rectangle((cx - 17, cy - 17, cx + 17, cy + 17), fill="#df3029", outline="#8b1511", width=4)
        draw.rectangle((bx - 52, by - 43, bx + 52, by + 43), fill="#8eb2f0", outline="#194eba", width=10)
        return image
