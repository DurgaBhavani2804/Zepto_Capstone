"""
Optional real-LLM path (MOCK_LLM=0 only). Not used, and not required, when
MOCK_LLM is left at its default - the mock branch in graph.py never imports
or calls anything in this file.

Uses Groq's free-tier API (https://console.groq.com) as the LLM backend.
Requires a GROQ_API_KEY environment variable when MOCK_LLM=0 is set.
"""

import json
import os
from typing import Optional

from pydantic import ValidationError

from schemas import AskResponse

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 2  # up to 2 additional attempts after the first, on validation failure


def _call_groq(prompt: str) -> str:
    """Single raw call to the Groq chat-completions API. Returns raw text."""
    from groq import Groq  # imported lazily so it is never required in mock mode

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires a GROQ_API_KEY environment variable "
            "(sign up free at https://console.groq.com)."
        )
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def generate_structured_answer(prompt: str) -> AskResponse:
    """
    Call the real LLM and validate its JSON output against AskResponse.
    On a validation failure, retry up to MAX_RETRIES more times with a
    corrective instruction appended to the prompt. If it still fails,
    return a clearly marked error response instead of raising.
    """
    last_error: Optional[Exception] = None
    current_prompt = prompt

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = _call_groq(current_prompt)
            data = json.loads(raw)
            return AskResponse(**data)
        except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
            last_error = exc
            current_prompt = (
                prompt
                + "\n\nYour previous reply did not match the required JSON schema "
                  "exactly (fields: answer: string, sources: list of strings, "
                  "confidence: float between 0 and 1). Reply again with ONLY a "
                  "valid JSON object matching that schema, and nothing else."
            )

    return AskResponse(
        answer=f"[error] Real-LLM output failed schema validation after "
               f"{MAX_RETRIES + 1} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )
