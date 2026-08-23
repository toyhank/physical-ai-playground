from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app.robot.backends import SO101MujocoBackend  # noqa: E402
from app.vla.controller import VLAController  # noqa: E402
from app.vla.mock import MockVLAProvider  # noqa: E402
from app.vla.smolvla_client import SmolVLAClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("mock", "smolvla"), default="mock")
    parser.add_argument("--host", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--checkpoint", default="ahmedsohail2003/smolvla-so101-pickplace-v2"
    )
    parser.add_argument("--dataset", default="none")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-mode", default="physics", choices=("physics", "contact_attachment"))
    parser.add_argument("--report-task-success", action="store_true")
    args = parser.parse_args()

    provider = MockVLAProvider() if args.provider == "mock" else SmolVLAClient(args.host)
    backend = SO101MujocoBackend(
        seed=args.seed,
        grasp_mode=args.grasp_mode,
        vla_aligned=args.provider == "smolvla",
    )
    controller = VLAController(backend, provider)
    tick = controller.tick("Pick up the red block and place it in the blue tray.")
    smoke = tick.result.success and len(tick.raw_action) == 6
    backend.stop()
    report: dict[str, object] = {
        "provider": args.provider,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "episode_count": args.episodes,
        "seed": args.seed,
        "grasp_mode": args.grasp_mode,
        "valid_action_chunk_smoke": smoke,
    }
    if not args.report_task_success:
        report["task_success_rate"] = None
        report["note"] = "Task success intentionally not reported without an explicit fine-tuned checkpoint benchmark."
        print(json.dumps(report, indent=2))
        return
    if args.provider != "smolvla" or args.checkpoint == "lerobot/smolvla_base" or args.dataset == "none":
        raise SystemExit("Refusing to report VLA success: provide a fine-tuned SmolVLA checkpoint and dataset.")
    successes = 0
    for episode in range(args.episodes):
        backend = SO101MujocoBackend(
            seed=args.seed + episode,
            grasp_mode=args.grasp_mode,
            vla_aligned=True,
        )
        controller = VLAController(backend, SmolVLAClient(args.host))
        for _ in range(600):
            result = controller.tick("Pick up the red block and place it in the blue tray.")
            if not result.result.success or backend.verify_task():
                break
        successes += int(backend.verify_task())
        backend.stop()
    report["successes"] = successes
    report["task_success_rate"] = successes / args.episodes
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
