from app.models.gemini_robotics import HYBRID_TOOLS


def test_hybrid_er2_tool_can_only_emit_language_subtask() -> None:
    assert len(HYBRID_TOOLS) == 1
    tool = HYBRID_TOOLS[0]
    assert tool["name"] == "execute_vla_subtask"
    properties = tool["parameters"]["properties"]
    assert set(properties) == {"instruction"}
    serialized = str(tool).lower()
    assert "joint_angle" not in serialized
    assert "cartesian" not in properties
    assert "object_id" not in properties
