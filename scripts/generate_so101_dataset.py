from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app.data.so101_dataset import SO101DemonstrationCollector, lerobot_v3_features  # noqa: E402
from app.robot.backends import SO101MujocoBackend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets" / "so101-mujoco-v3")
    parser.add_argument("--repo-id", default="local/so101-mujoco-v3")
    args = parser.parse_args()

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise SystemExit('Install dataset tooling with: pip install "lerobot>=0.4,<0.6"') from error

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=30,
        root=args.root,
        robot_type="so101_mujoco",
        features=lerobot_v3_features(),
        use_videos=True,
        image_writer_threads=4,
    )
    collector = SO101DemonstrationCollector()
    saved = 0
    attempts = 0
    while saved < args.episodes and attempts < args.episodes * 10:
        backend = SO101MujocoBackend(seed=args.seed + attempts, grasp_mode="physics")
        episode = collector.collect(backend)
        backend.stop()
        attempts += 1
        if not episode.success:
            print(f"discard attempt={attempts} error={episode.error}")
            continue
        for frame in episode.frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1
        print(f"saved episode {saved}/{args.episodes} ({len(episode.frames)} frames)")
    dataset.finalize()
    if saved < args.episodes:
        raise SystemExit(f"Only collected {saved}/{args.episodes} successful episodes")


if __name__ == "__main__":
    main()
