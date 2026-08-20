from __future__ import annotations

import asyncio

from app.agent.orchestrator import AgentOrchestrator
from app.config import Settings
from app.models.mock import MockRobotModel
from app.simulation.mujoco_engine import MujocoEngine
from app.sessions.session import SimulationSession


def test_mock_agent_is_language_agnostic() -> None:
    engine = MujocoEngine(seed=7, enable_mujoco=False)
    provider = MockRobotModel()
    chinese = provider.plan("把红色方块放进蓝色盒子", engine)
    english = provider.plan("Put the red cube into the blue box.", engine)
    assert chinese == english
    assert [action["name"] for action in chinese] == ["pick_object", "place_object"]


def test_mock_agent_end_to_end_is_verified_by_simulator() -> None:
    async def scenario() -> None:
        session = SimulationSession(seed=41)
        queue = session.subscribe()
        orchestrator = AgentOrchestrator(Settings(model_provider="mock"))
        await orchestrator.run(session, "把红色方块移到蓝色盒子里")
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert session.engine.verify_task()
        assert len([event for event in events if event["type"] == "tool"]) == 2
        assert any(event["type"] == "success" and event["verified"] for event in events)

    asyncio.run(scenario())
