from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.robot.so101_spec import SO101_JOINTS


@dataclass(frozen=True)
class FeatureMetadata:
    """Normalization metadata supplied by a policy checkpoint/dataset."""

    mode: Literal["identity", "min_max", "mean_std"] = "identity"
    minimum: tuple[float, ...] | None = None
    maximum: tuple[float, ...] | None = None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    normalized_min: float = -1.0
    normalized_max: float = 1.0


class ActionCodec:
    dimension = SO101_JOINTS.size

    def __init__(
        self,
        *,
        action: FeatureMetadata | None = None,
        state: FeatureMetadata | None = None,
    ) -> None:
        self.action_metadata = action or FeatureMetadata()
        self.state_metadata = state or FeatureMetadata()

    def decode_policy_action(self, value: np.ndarray | list[float]) -> np.ndarray:
        array = self._validate(value)
        decoded = self._decode(array, self.action_metadata)
        lower = np.asarray(SO101_JOINTS.lower, dtype=np.float32)
        upper = np.asarray(SO101_JOINTS.upper, dtype=np.float32)
        if np.any(decoded < lower) or np.any(decoded > upper):
            raise ValueError("DECODED_ACTION_OUTSIDE_JOINT_LIMITS")
        return decoded.astype(np.float32)

    def encode_robot_state(self, value: np.ndarray | list[float]) -> np.ndarray:
        array = self._validate(value)
        return self._encode(array, self.state_metadata).astype(np.float32)

    def _validate(self, value: np.ndarray | list[float]) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1:] != (self.dimension,):
            raise ValueError("SO101_FEATURE_SHAPE_MISMATCH")
        if not np.all(np.isfinite(array)):
            raise ValueError("SO101_FEATURE_NON_FINITE")
        return array

    def _decode(self, value: np.ndarray, metadata: FeatureMetadata) -> np.ndarray:
        if metadata.mode == "identity":
            return value.copy()
        if metadata.mode == "min_max":
            minimum, maximum = self._pair(metadata.minimum, metadata.maximum)
            span = metadata.normalized_max - metadata.normalized_min
            if span <= 0:
                raise ValueError("INVALID_NORMALIZED_RANGE")
            ratio = (value - metadata.normalized_min) / span
            return minimum + ratio * (maximum - minimum)
        mean, std = self._pair(metadata.mean, metadata.std)
        if np.any(std <= 0):
            raise ValueError("INVALID_STANDARD_DEVIATION")
        return value * std + mean

    def _encode(self, value: np.ndarray, metadata: FeatureMetadata) -> np.ndarray:
        if metadata.mode == "identity":
            return value.copy()
        if metadata.mode == "min_max":
            minimum, maximum = self._pair(metadata.minimum, metadata.maximum)
            span = maximum - minimum
            if np.any(span <= 0):
                raise ValueError("INVALID_FEATURE_RANGE")
            ratio = (value - minimum) / span
            return metadata.normalized_min + ratio * (
                metadata.normalized_max - metadata.normalized_min
            )
        mean, std = self._pair(metadata.mean, metadata.std)
        if np.any(std <= 0):
            raise ValueError("INVALID_STANDARD_DEVIATION")
        return (value - mean) / std

    def _pair(
        self,
        first: tuple[float, ...] | None,
        second: tuple[float, ...] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if first is None or second is None:
            raise ValueError("NORMALIZATION_METADATA_MISSING")
        a = np.asarray(first, dtype=np.float32)
        b = np.asarray(second, dtype=np.float32)
        if a.shape != (self.dimension,) or b.shape != (self.dimension,):
            raise ValueError("NORMALIZATION_METADATA_SHAPE_MISMATCH")
        return a, b
