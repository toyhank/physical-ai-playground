from __future__ import annotations

import os

import pytest


@pytest.mark.gemini
def test_gemini_access_is_explicitly_opt_in() -> None:
    if os.getenv("RUN_GEMINI_TESTS") != "1":
        pytest.skip("Set RUN_GEMINI_TESTS=1 to run paid/external Gemini tests")
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-robotics-er-2-preview",
        contents="Reply exactly MODEL ACCESS OK",
    )
    assert "MODEL ACCESS OK" in (response.text or "")

