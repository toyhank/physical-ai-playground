"""Smoke-test Gemini Robotics ER 2 without robot hardware or MuJoCo."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors
from google.genai._gaos.lib.compat_errors import APIStatusError as InteractionsAPIError
from PIL import Image, ImageDraw


MODEL_ID = "gemini-robotics-er-2-preview"
SCENE_PATH = Path(__file__).with_name("test_scene.png")
MAX_AGENT_STEPS = 15


MOVE_TOOL = {
    "type": "function",
    "name": "move",
    "description": (
        "Move the robot gripper to normalized image coordinates. "
        "x and y are integers from 0 to 1000. Set high=true to stay safely "
        "above the table, or high=false to descend to table level."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Normalized horizontal image coordinate.",
            },
            "y": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Normalized vertical image coordinate.",
            },
            "high": {
                "type": "boolean",
                "description": "True for above the table; false for table level.",
            },
        },
        "required": ["x", "y", "high"],
    },
}

GRIPPER_TOOL = {
    "type": "function",
    "name": "set_gripper_state",
    "description": "Open or close the robot gripper.",
    "parameters": {
        "type": "object",
        "properties": {
            "opened": {
                "type": "boolean",
                "description": "True opens the gripper; false closes it.",
            }
        },
        "required": ["opened"],
    },
}

TOOLS = [MOVE_TOOL, GRIPPER_TOOL]


def create_test_image() -> str:
    """Create a deterministic 1000x1000 scene and return its PNG as base64."""
    image = Image.new("RGB", (1000, 1000), "#eeeeea")
    draw = ImageDraw.Draw(image)

    # Red block center: (x=293, y=557).
    draw.rectangle((233, 497, 353, 617), fill="#d92f2f", outline="#781515", width=6)

    # Blue open container center: (x=752, y=547).
    draw.rounded_rectangle(
        (642, 447, 862, 647), radius=24, fill="#2e67c7", outline="#15356e", width=8
    )
    draw.rounded_rectangle(
        (672, 477, 832, 617), radius=16, fill="#b9d4ff", outline="#15356e", width=6
    )

    image.save(SCENE_PATH)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_input(img_b64: str, prompt: str) -> list[dict[str, Any]]:
    """Build the current Interactions API multimodal user input."""
    return [
        {
            "type": "user_input",
            "content": [
                {
                    "type": "image",
                    "data": img_b64,
                    "mime_type": "image/png",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def test_spatial_reasoning(client: genai.Client, img_b64: str) -> None:
    print("\n=== TEST A: SPATIAL REASONING ===")
    prompt = """
Find the red block and the blue container in the image.
Return ONLY valid JSON in exactly this shape:
{
  "red_block": {"point": [y, x]},
  "blue_container": {"point": [y, x]}
}
Coordinates must use [y, x] order and be normalized to integers from 0 to 1000.
Point to the center of each object.
""".strip()

    interaction = client.interactions.create(
        model=MODEL_ID,
        input=image_input(img_b64, prompt),
        generation_config={"thinking_level": "low"},
    )

    print(interaction.output_text or "(No text output)")
    print("Expected approximately:")
    print(
        json.dumps(
            {
                "red_block": {"point": [557, 293]},
                "blue_container": {"point": [547, 752]},
            },
            indent=2,
        )
    )


def validate_call(name: str, arguments: dict[str, Any]) -> None:
    """Reject malformed mock calls so invalid plans cannot look successful."""
    if name == "move":
        if set(arguments) != {"x", "y", "high"}:
            raise ValueError(f"move received unexpected arguments: {arguments}")
        if not all(isinstance(arguments[key], int) for key in ("x", "y")):
            raise ValueError(f"move coordinates must be integers: {arguments}")
        if not all(0 <= arguments[key] <= 1000 for key in ("x", "y")):
            raise ValueError(f"move coordinates are outside 0..1000: {arguments}")
        if not isinstance(arguments["high"], bool):
            raise ValueError(f"move.high must be boolean: {arguments}")
        return

    if name == "set_gripper_state":
        if set(arguments) != {"opened"} or not isinstance(arguments["opened"], bool):
            raise ValueError(f"invalid gripper arguments: {arguments}")
        return

    raise ValueError(f"model requested unknown tool: {name}")


def test_robot_orchestration(client: genai.Client, img_b64: str) -> None:
    print("\n=== TEST B: ROBOT ORCHESTRATION (MOCK ONLY) ===")
    prompt = """
You control a simulated top-down robot arm. Inspect the image, then pick up the
red block and place it inside the blue container. Coordinates are normalized
image coordinates from 0 to 1000, with x horizontal and y vertical.

Use the supplied functions to perform the complete safe sequence. Start with an
open gripper. Always approach and leave an object with high=true. Descend with
high=false only when grasping or releasing. Do not merely describe the plan:
call the functions in the required order. The functions are mocked and always
succeed; no real robot is connected.
""".strip()

    interaction = client.interactions.create(
        model=MODEL_ID,
        input=image_input(img_b64, prompt),
        tools=TOOLS,
        generation_config={"thinking_level": "low"},
    )

    for _ in range(MAX_AGENT_STEPS):
        calls = [step for step in interaction.steps if step.type == "function_call"]

        if not calls:
            print("\nMODEL FINISHED")
            if interaction.output_text:
                print("Model:", interaction.output_text)
            return

        results = []
        for call in calls:
            arguments = dict(call.arguments)
            validate_call(call.name, arguments)

            print(f"\nTOOL CALL: {call.name}")
            print(json.dumps(arguments, indent=2, ensure_ascii=False))

            # Smoke test only: acknowledge the call without touching hardware.
            results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps({"status": "success"}),
                        }
                    ],
                }
            )

        interaction = client.interactions.create(
            model=MODEL_ID,
            previous_interaction_id=interaction.id,
            tools=TOOLS,
            input=results,
            generation_config={"thinking_level": "low"},
        )

    raise RuntimeError(f"Exceeded maximum agent steps ({MAX_AGENT_STEPS}).")


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it only in your local PowerShell "
            "session; do not paste it into chat or source code."
        )

    client = genai.Client(api_key=api_key)
    img_b64 = create_test_image()

    print(f"Generated {SCENE_PATH}")
    print("Using model:", MODEL_ID)
    print("No robot hardware is connected; all tool results are simulated.")

    test_spatial_reasoning(client, img_b64)
    test_robot_orchestration(client, img_b64)


def api_error_code(error: Exception) -> int | None:
    """Read the status code from either google-genai API error family."""
    return getattr(error, "code", None) or getattr(error, "status_code", None)


def explain_api_error(error: Exception) -> str:
    """Turn common API failures into an actionable smoke-test diagnosis."""
    code = api_error_code(error)
    if code == 400 and "current location" in str(error).lower():
        return (
            "Google rejected the request's detected network location. Confirm "
            "that the actual egress country is on Google's supported-region list."
        )
    diagnoses = {
        400: "Bad request. Inspect the API message for the exact reason.",
        401: "API key is invalid. Create/check the key in Google AI Studio.",
        403: "The key, project, or current region does not have model access.",
        404: f"Model not found. Confirm the exact model ID: {MODEL_ID}",
        429: "Quota or rate limit reached; this does not mean the model is unavailable.",
    }
    return diagnoses.get(code, "Unexpected Gemini API failure; inspect the details below.")


if __name__ == "__main__":
    try:
        main()
    except (errors.APIError, InteractionsAPIError) as error:
        code = api_error_code(error)
        status = getattr(error, "status", None) or "unknown status"
        print(f"\nAPI ERROR {code} ({status})")
        print(explain_api_error(error))
        print("Details:", getattr(error, "message", None) or str(error))
        raise SystemExit(1) from None
