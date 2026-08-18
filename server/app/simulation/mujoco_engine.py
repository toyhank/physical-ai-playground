from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.robot.ik import PlanarDampedLeastSquaresIK
from app.simulation.camera import FixedCameraProjector


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
                    <site name="gripper_site" pos="0.07 0 0" size="0.018" rgba="0.8 1 0.3 1"/>
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
</mujoco>
"""


@dataclass
class ToolExecution:
    success: bool
    error: str | None = None
    frames: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.error:
            result["error"] = self.error
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
        self._renderer = None
        if enable_mujoco:
            try:
                import mujoco

                self._mujoco = mujoco
                self.model = mujoco.MjModel.from_xml_string(MJCF)
                self.data = mujoco.MjData(self.model)
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
        self._sync_mujoco()
        return self.state()

    def _sync_mujoco(self) -> None:
        if not self.mujoco_enabled:
            return
        self.data.qpos[:7] = self.robot_joints
        cube_joint = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free")
        cube_qpos = self.model.jnt_qposadr[cube_joint]
        self.data.qpos[cube_qpos : cube_qpos + 3] = self.cube_position
        self.data.qpos[cube_qpos + 3 : cube_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.mocap_pos[0] = self.box_position
        self._mujoco.mj_forward(self.model, self.data)

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
            self.robot_joints = start_joints + fraction * (end_joints - start_joints)
            if self.grasped:
                self.cube_position = self.ee_position + np.asarray((0.0, 0.0, -0.04))
            self.frame += 1
            self._sync_mujoco()
            frames.append(self.state())
        return ToolExecution(True, frames=frames)

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
                self._sync_mujoco()
                return ToolExecution(False, "GRASP_FAILED")
            self.grasped = True
        self.frame += 1
        self._sync_mujoco()
        return ToolExecution(True, frames=[self.state()])

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
                if self._renderer is None:
                    self._renderer = self._mujoco.Renderer(self.model, height=512, width=512)
                self._renderer.update_scene(self.data, camera="robot_camera")
                pixels = self._renderer.render()
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

