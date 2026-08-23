from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app.robot.backends import SO101MujocoBackend  # noqa: E402
from app.robot.expert import SO101ExpertPolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-mode", choices=("physics", "contact_attachment"), default="physics")
    args = parser.parse_args()

    failures: Counter[str] = Counter()
    steps: list[int] = []
    durations: list[float] = []
    successes = 0
    for episode in range(args.episodes):
        backend = SO101MujocoBackend(seed=args.seed + episode, grasp_mode=args.grasp_mode)
        started = time.perf_counter()
        result = SO101ExpertPolicy(backend).run()
        durations.append(time.perf_counter() - started)
        steps.append(backend.simulation_steps)
        successes += int(result.success)
        if not result.success:
            category = {
                "IK_FAILED": "ik_failure",
                "PHYSICS_GRASP_FAILED": "grasp_failure",
                "PLACE_VERIFICATION_FAILED": "place_failure",
            }.get(result.error or "", "slip_or_drop")
            failures[category] += 1
        backend.stop()

    report = {
        "episodes": args.episodes,
        "seed": args.seed,
        "grasp_mode": args.grasp_mode,
        "successes": successes,
        "success_rate": successes / args.episodes if args.episodes else 0.0,
        "grasp_failure": failures["grasp_failure"],
        "ik_failure": failures["ik_failure"],
        "slip_or_drop": failures["slip_or_drop"],
        "place_failure": failures["place_failure"],
        "mean_physics_steps": statistics.fmean(steps) if steps else 0.0,
        "mean_episode_seconds": statistics.fmean(durations) if durations else 0.0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
