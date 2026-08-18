"""Run the deterministic pick-and-place benchmark without external APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app.simulation.mujoco_engine import MujocoEngine  # noqa: E402


def main() -> None:
    episodes = 100
    successes = 0
    physics = "unknown"
    for seed in range(episodes):
        engine = MujocoEngine(seed=seed)
        physics = engine.state()["physics"]
        engine.deterministic_pick_place()
        successes += int(engine.verify_task())

    result = {
        "provider": "mock",
        "physics": physics,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if successes >= 95 else 1)


if __name__ == "__main__":
    main()
