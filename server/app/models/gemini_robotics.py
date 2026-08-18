from __future__ import annotations

import json
import os
from typing import Any

from google import genai

from app.agent.prompt import ROBOT_SYSTEM_PROMPT


MODEL_ID = "gemini-robotics-er-2-preview"
TOOLS = [
    {
        "type": "function",
        "name": "move",
        "description": "Move the gripper to normalized image coordinates; high means safely above the table.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                "high": {"type": "boolean"},
            },
            "required": ["x", "y", "high"],
        },
    },
    {
        "type": "function",
        "name": "set_gripper_state",
        "description": "Open or close the robot gripper.",
        "parameters": {
            "type": "object",
            "properties": {"opened": {"type": "boolean"}},
            "required": ["opened"],
        },
    },
]


class GeminiRoboticsProvider:
    name = "gemini-robotics-er-2"

    def __init__(self, api_key: str | None = None, model_id: str = MODEL_ID) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required when MODEL_PROVIDER=gemini")
        self.client = genai.Client(api_key=key)
        self.model_id = model_id

    @staticmethod
    def _image_input(image_b64: str, text: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "user_input",
                "content": [
                    {"type": "image", "data": image_b64, "mime_type": "image/png"},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def start(self, prompt: str, image_b64: str) -> Any:
        return self.client.interactions.create(
            model=self.model_id,
            system_instruction=ROBOT_SYSTEM_PROMPT,
            input=self._image_input(image_b64, prompt),
            tools=TOOLS,
            generation_config={"thinking_level": "low"},
        )

    @staticmethod
    def calls(interaction: Any) -> list[Any]:
        return [step for step in interaction.steps if step.type == "function_call"]

    def continue_after_tools(
        self, interaction: Any, tool_results: list[tuple[Any, dict[str, Any]]], image_b64: str
    ) -> Any:
        inputs = []
        for index, (call, result) in enumerate(tool_results):
            result_parts: list[dict[str, Any]] = [
                {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
            ]
            if index == len(tool_results) - 1:
                result_parts.append({"type": "image", "data": image_b64, "mime_type": "image/png"})
            inputs.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": result_parts,
                }
            )
        return self.client.interactions.create(
            model=self.model_id,
            previous_interaction_id=interaction.id,
            system_instruction=ROBOT_SYSTEM_PROMPT,
            tools=TOOLS,
            input=inputs,
            generation_config={"thinking_level": "low"},
        )

