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
        if session.controller == "hybrid":
            await self._run_hybrid(session, prompt, logger)
            return
        if session.controller == "vla":
            await self._run_vla(session, prompt, logger)
            return
        if session.brain == "none":
            await self._run_mock(session, prompt, logger)
            return
        if session.brain == "gemini":
            await self._run_gemini(session, prompt, logger)
            return
        raise RuntimeError(f"Unsupported brain: {session.brain}")

    async def _run_vla(
        self, session: SimulationSession, prompt: str, logger: RunLogger
    ) -> None:
        from app.robot.backends.so101_mujoco import SO101MujocoBackend
        from app.vla.controller import VLAController
        from app.vla.mock import MockVLAProvider
        from app.vla.smolvla_client import SmolVLAClient

        if not isinstance(session.backend, SO101MujocoBackend):
            raise RuntimeError("VLA_REQUIRES_SO101")
        if self.config.vla_provider == "mock":
            provider = MockVLAProvider()
        elif self.config.vla_provider == "smolvla":
            provider = SmolVLAClient(self.config.vla_host)
        else:
            raise RuntimeError("UNKNOWN_VLA_PROVIDER")
        controller = VLAController(session.backend, provider)
        policy_instruction = self.config.vla_task_prompt
        await session.publish(
            {
                "type": "vla_observation",
                "features": [
                    "observation.images.scene",
                    "observation.images.wrist",
                    "observation.state",
                ],
                "forbidden_features": ["object_xyz", "target_object_id", "ik_target"],
            }
        )
        logger.event(
            {
                "type": "vla",
                "provider": provider.name,
                "requested_task": prompt,
                "policy_instruction": policy_instruction,
            }
        )
        for _ in range(self.config.vla_max_steps):
            tick = await asyncio.to_thread(controller.tick, policy_instruction)
            event = {
                "type": "vla_action",
                "provider": provider.name,
                "raw_action": tick.raw_action,
                "safe_action": tick.safe_action,
                "safety": tick.safety_events,
                "action_queue": tick.queue_size,
                "inference_latency_ms": round(controller.last_inference_ms, 3),
            }
            logger.event(event)
            await session.publish(event)
            for safety_event in tick.safety_events:
                await session.publish({"type": "safety", "event": safety_event})
            for frame in tick.result.frames or []:
                await session.publish({"type": "scene_state", "state": frame})
            if not tick.result.success:
                await session.publish({"type": "failure", "error": tick.result.error})
                logger.finish(False, tick.result.error)
                return
            if session.backend.verify_task():
                break
            await asyncio.sleep(1 / session.backend.policy_hz)
        await self._finish(session, logger)

    async def _execute_vla_subtask(
        self,
        session: SimulationSession,
        instruction: str,
        logger: RunLogger,
    ) -> dict[str, Any]:
        from app.robot.backends.so101_mujoco import SO101MujocoBackend
        from app.vla.controller import VLAController
        from app.vla.mock import MockVLAProvider
        from app.vla.smolvla_client import SmolVLAClient

        if not isinstance(session.backend, SO101MujocoBackend):
            return {"success": False, "error": "VLA_REQUIRES_SO101"}
        provider = (
            MockVLAProvider()
            if self.config.vla_provider == "mock"
            else SmolVLAClient(self.config.vla_host)
        )
        controller = VLAController(session.backend, provider)
        policy_instruction = self.config.vla_task_prompt
        await session.publish(
            {
                "type": "vla_subtask",
                "instruction": instruction,
                "policy_instruction": policy_instruction,
                "provider": provider.name,
            }
        )
        for _ in range(self.config.vla_max_steps):
            tick = await asyncio.to_thread(controller.tick, policy_instruction)
            event = {
                "type": "vla_action",
                "provider": provider.name,
                "raw_action": tick.raw_action,
                "safe_action": tick.safe_action,
                "safety": tick.safety_events,
                "action_queue": tick.queue_size,
                "inference_latency_ms": round(controller.last_inference_ms, 3),
            }
            logger.event(event)
            await session.publish(event)
            for frame in tick.result.frames or []:
                await session.publish({"type": "scene_state", "state": frame})
            if not tick.result.success:
                return {"success": False, "error": tick.result.error}
            if session.backend.verify_task():
                return {"success": True, "verified": True}
            await asyncio.sleep(1 / session.backend.policy_hz)
        return {"success": False, "error": "VLA_SUBTASK_STEP_LIMIT"}

    async def _run_hybrid(
        self, session: SimulationSession, prompt: str, logger: RunLogger
    ) -> None:
        from app.models.gemini_robotics import GeminiRoboticsProvider

        if session.brain != "gemini":
            raise RuntimeError("HYBRID_REQUIRES_GEMINI_BRAIN")
        provider = GeminiRoboticsProvider(model_id=self.config.model_id, hybrid=True)
        scene_b64, wrist_b64 = await self._observe(session, logger, 0)
        interaction = await asyncio.to_thread(provider.start, prompt, scene_b64, wrist_b64)
        observation = 0
        while observation < self.config.max_agent_steps:
            calls = provider.calls(interaction)
            if not calls:
                break
            results: list[tuple[Any, dict[str, Any]]] = []
            for call in calls:
                if call.name != "execute_vla_subtask":
                    result = {"success": False, "error": "HYBRID_TOOL_NOT_ALLOWED"}
                else:
                    instruction = str(dict(call.arguments).get("instruction", "")).strip()
                    if not instruction:
                        result = {"success": False, "error": "EMPTY_VLA_SUBTASK"}
                    else:
                        result = await self._execute_vla_subtask(
                            session, instruction, logger
                        )
                results.append((call, result))
            observation += 1
            scene_b64, wrist_b64 = await self._observe(
                session, logger, observation
            )
            interaction = await asyncio.to_thread(
                provider.continue_after_tools,
                interaction,
                results,
                scene_b64,
                wrist_b64,
            )
            if session.backend.verify_task():
                break
        await self._finish(session, logger)

    async def _observe(
        self,
        session: SimulationSession,
        logger: RunLogger,
        index: int,
        *,
        latch_projection: bool = True,
    ) -> tuple[str, str]:
        scene_b64 = await asyncio.to_thread(
            session.backend.render_camera,
            "scene",
            latch_observation=latch_projection,
        )
        wrist_b64 = await asyncio.to_thread(
            session.backend.render_camera,
            "wrist",
            latch_observation=False,
        )
        logger.observation(index, base64.b64decode(scene_b64), "scene")
        logger.observation(index, base64.b64decode(wrist_b64), "wrist")
        event = {
            "type": "observe",
            "text": "Captured fixed scene + wrist cameras",
            "observation": index,
        }
        logger.event(event)
        await session.publish(event)
        return scene_b64, wrist_b64

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
        verified = session.backend.verify_task()
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
        actions = provider.plan(prompt, session.backend)
        await session.publish({"type": "model", "text": "Visual task plan ready", "model": provider.name})
        observation = 0
        for action in actions:
            result = await self._execute_action(
                session, logger, action["name"], action["arguments"]
            )
            if session.backend.verify_task():
                break
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
        scene_b64, wrist_b64 = await self._observe(session, logger, 0)
        interaction = await asyncio.to_thread(
            provider.start, prompt, scene_b64, wrist_b64
        )
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
            if session.backend.verify_task():
                break
            observation += 1
            scene_b64, wrist_b64 = await self._observe(session, logger, observation)
            interaction = await asyncio.to_thread(
                provider.continue_after_tools,
                interaction,
                results,
                scene_b64,
                wrist_b64,
            )
        await self._finish(session, logger)
