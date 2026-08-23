import numpy as np

from app.data.so101_dataset import TRAINING_FEATURE_KEYS, build_training_frame, lerobot_v3_features


def test_training_frame_whitelists_only_vla_features() -> None:
    observation = {
        "observation.images.scene": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
        "object_xyz": [1, 2, 3],
        "ik_target": [4, 5, 6],
    }
    frame = build_training_frame(observation, np.zeros(6), "test task")
    assert set(frame) == TRAINING_FEATURE_KEYS
    assert "object_xyz" not in frame
    assert lerobot_v3_features()["action"]["shape"] == (6,)
