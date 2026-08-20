from __future__ import annotations

import asyncio
import base64
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.models.mock import MockRobotModel
from app.run_logging.run_logger import RunLogger
from app.sessions.session import SimulationSession


class AgentOrchestrator:
    def __init__(self, config: Settings) -> None:
        self.config = config

    async def run(self, session: SimulationSession, prompt: str) -> None:
        run_id = uuid4().hex
        logger = RunLogger(run_id, prompt, self.config.model_provider)
        await session.publish({"type": "run_started", "run_id": run_id, "prompt": prompt})
        logger.event({"type": "user", "text": prompt})
        try:
            await asyncio.wait_for(
                self._run_provider(session, prompt, logger), timeout=self.config.max_task_seconds
            )
        except TimeoutError:
            logger.finish(False, "TASK_TIMEOUT")
            await session.publish({"type": "failure", "error": "TASK_TIMEOUT"})
        except asyncio.CancelledError:
            logger.finish(False, "STOPPED_BY_USER")
            await session.publish({"type": "stopped", "error": "STOPPED_BY_USER"})
            raise
        except Exception as error:
            logger.finish(False, type(error).__name__)
            await session.publish({"type": "failure", "error": str(error)})

    async def _run_provider(self, session: SimulationSession, prompt: str, logger: RunLogger) -> None:
        if self.config.model_provider == "mock":
            await self._run_mock(session, prompt, logger)
            return
        if self.config.model_provider == "gemini":
            await self._run_gemini(session, prompt, logger)
            return
        raise RuntimeError(f"Unsupported MODEL_PROVIDER: {self.config.model_provider}")

    async def _observe(
        self,
        session: SimulationSession,
        logger: RunLogger,
        index: int,
        *,
        latch_projection: bool = True,
    ) -> str:
        image_b64 = await asyncio.to_thread(
            session.engine.camera_png_base64,
            latch_observation=latch_projection,
        )
        logger.observation(index, base64.b64decode(image_b64))
        event = {"type": "observe", "text": "Captured robot camera", "observation": index}
        logger.event(event)
        await session.publish(event)
        return image_b64

    async def _execute_action(
        self,
        session: SimulationSession,
        logger: RunLogger,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        tool_event = {"type": "tool", "name": name, "arguments": arguments}
        logger.event(tool_event)
        await session.publish(tool_event)
        result = await asyncio.to_thread(session.tools.execute, name, arguments)
        for frame in result.frames or []:
            await session.publish({"type": "scene_state", "state": frame})
            await asyncio.sleep(0.025)
        payload = result.as_dict()
        logger.event({"type": "result", "name": name, "result": payload})
        await session.publish({"type": "result", "name": name, "result": payload})
        return payload

    async def _finish(self, session: SimulationSession, logger: RunLogger) -> None:
        verified = session.engine.verify_task()
        if verified:
            logger.finish(True)
            await session.publish(
                {"type": "success", "text": "Verified by simulator", "verified": True}
            )
        else:
            logger.finish(False, "SIMULATOR_VERIFICATION_FAILED")
            await session.publish(
                {
                    "type": "failure",
                    "error": "SIMULATOR_VERIFICATION_FAILED",
                    "verified": False,
                }
            )

    async def _run_mock(self, session: SimulationSession, prompt: str, logger: RunLogger) -> None:
        await self._observe(session, logger, 0)
        provider = MockRobotModel()
        actions = provider.plan(prompt, session.engine)
        await session.publish({"type": "model", "text": "Visual task plan ready", "model": provider.name})
        observation = 0
        for action in actions:
            result = await self._execute_action(
                session, logger, action["name"], action["arguments"]
            )
            observation += 1
            # The mock planner creates one complete batch from observation 0.
            # Show the moving wrist camera without changing that batch's pixel frame.
            await self._observe(session, logger, observation, latch_projection=False)
            if not result["success"]:
                await session.publish(
                    {"type": "model", "text": "Tool failed; re-observing before retry"}
                )
                break
        await self._finish(session, logger)

    async def _run_gemini(self, session: SimulationSession, prompt: str, logger: RunLogger) -> None:
        from app.models.gemini_robotics import GeminiRoboticsProvider

        provider = GeminiRoboticsProvider(model_id=self.config.model_id)
        image_b64 = await self._observe(session, logger, 0)
        interaction = await asyncio.to_thread(provider.start, prompt, image_b64)
        observation = 0
        while session.tools.steps < self.config.max_agent_steps:
            calls = provider.calls(interaction)
            if not calls:
                if interaction.output_text:
                    await session.publish(
                        {"type": "model", "text": interaction.output_text, "model": provider.name}
                    )
                break
            results = []
            for call in calls:
                arguments = dict(call.arguments)
                result = await self._execute_action(session, logger, call.name, arguments)
                results.append((call, result))
                if session.tools.steps >= self.config.max_agent_steps:
                    break
            observation += 1
            image_b64 = await self._observe(session, logger, observation)
            interaction = await asyncio.to_thread(
                provider.continue_after_tools, interaction, results, image_b64
            )
        await self._finish(session, logger)
