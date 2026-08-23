import numpy as np
import pytest

from app.robot.so101_spec import SO101_JOINTS
from app.vla.action_codec import ActionCodec, FeatureMetadata


def test_metadata_driven_min_max_round_trip() -> None:
    limits = FeatureMetadata(
        mode="min_max",
        minimum=SO101_JOINTS.lower,
        maximum=SO101_JOINTS.upper,
    )
    codec = ActionCodec(action=limits, state=limits)
    midpoint = (np.asarray(SO101_JOINTS.lower) + np.asarray(SO101_JOINTS.upper)) / 2
    assert np.allclose(codec.encode_robot_state(midpoint), 0.0, atol=1e-6)
    assert np.allclose(codec.decode_policy_action(np.zeros(6)), midpoint, atol=1e-6)


def test_codec_rejects_missing_metadata_and_bad_gripper_range() -> None:
    with pytest.raises(ValueError, match="NORMALIZATION_METADATA_MISSING"):
        ActionCodec(action=FeatureMetadata(mode="min_max")).decode_policy_action(np.zeros(6))
    invalid = np.zeros(6, dtype=np.float32)
    invalid[-1] = 9.0
    with pytest.raises(ValueError, match="JOINT_LIMITS"):
        ActionCodec().decode_policy_action(invalid)
