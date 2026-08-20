from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedCameraProjector:
    """Pinhole camera projection and table-plane ray intersection."""

    position: tuple[float, float, float] = (0.0, -1.15, 1.35)
    target: tuple[float, float, float] = (0.0, 0.05, 0.44)
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_fov_degrees: float = 49.0
    width: int = 512
    height: int = 512
    table_z: float = 0.44

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin = np.asarray(self.position, dtype=float)
        forward = np.asarray(self.target, dtype=float) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(self.up_hint, dtype=float))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return right, up, forward

    @property
    def focal_pixels(self) -> float:
        return self.height / (2.0 * np.tan(np.deg2rad(self.vertical_fov_degrees) / 2.0))

    def normalized_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return x * self.width / 1000.0, y * self.height / 1000.0

    def pixel_to_normalized(self, px: float, py: float) -> tuple[int, int]:
        return round(px * 1000.0 / self.width), round(py * 1000.0 / self.height)

    def image_point_to_world(
        self,
        x: float,
        y: float,
        *,
        position: np.ndarray | tuple[float, float, float] | None = None,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        origin, camera_ray = self.image_ray(x, y, position=position, basis=basis)
        if abs(camera_ray[2]) < 1e-9:
            raise ValueError("camera ray is parallel to the table")
        distance = (self.table_z - origin[2]) / camera_ray[2]
        if distance <= 0:
            raise ValueError("camera ray does not intersect the table in front of the camera")
        return origin + distance * camera_ray

    def image_ray(
        self,
        x: float,
        y: float,
        *,
        position: np.ndarray | tuple[float, float, float] | None = None,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            raise ValueError("normalized coordinates must be within 0..1000")
        px, py = self.normalized_to_pixel(x, y)
        right, up, forward = basis if basis is not None else self._basis()
        camera_ray = (
            forward
            + ((px - self.width / 2.0) / self.focal_pixels) * right
            - ((py - self.height / 2.0) / self.focal_pixels) * up
        )
        camera_ray /= np.linalg.norm(camera_ray)
        origin = np.asarray(self.position if position is None else position, dtype=float)
        return origin, camera_ray

    def world_to_normalized(
        self,
        point: np.ndarray | tuple[float, float, float],
        *,
        position: np.ndarray | tuple[float, float, float] | None = None,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[int, int]:
        world = np.asarray(point, dtype=float)
        origin = np.asarray(self.position if position is None else position, dtype=float)
        right, up, forward = basis if basis is not None else self._basis()
        relative = world - origin
        depth = float(np.dot(relative, forward))
        if depth <= 0:
            raise ValueError("point is behind the camera")
        px = self.width / 2.0 + self.focal_pixels * float(np.dot(relative, right)) / depth
        py = self.height / 2.0 - self.focal_pixels * float(np.dot(relative, up)) / depth
        return self.pixel_to_normalized(px, py)
