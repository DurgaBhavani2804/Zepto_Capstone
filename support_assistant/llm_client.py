"""
Real-LLM client used ONLY on the optional MOCK_LLM=0 extension path.

This module is never imported/called when MOCK_LLM is left at its default
(unset or "1") — the mock nodes in graph.py build answers directly in code.

Uses Groq's free-tier API (OpenAI-compatible /chat/completions endpoint) by
default. Any other LLM API with a genuinely free tier can be substituted by
changing GROQ_API_URL / GROQ_MODEL / the auth header below.
"""

import json
import os
import re
import urllib.error
import urllib.request

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


class LLMError(Exception):
    pass


def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Send `prompt` to the configured LLM and return the raw text response.

    Only called when MOCK_LLM=0. Requires GROQ_API_KEY to be set in the
    environment (e.g. as a Space secret when deployed, or a local env var).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError(
            "MOCK_LLM=0 was set but GROQ_API_KEY is not configured. "
            "Set MOCK_LLM=1 (or leave it unset) to use the graded offline mock path, "
            "or set GROQ_API_KEY to use the optional real-LLM extension."
        )

    body = json.dumps(
        {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        GROQ_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise LLMError(f"LLM API call failed: {e}") from e

    return data["choices"][0]["message"]["content"]


def extract_json(raw_text: str) -> dict:
    """Best-effort extraction of a JSON object from raw LLM text
    (strips markdown code fences if the model added them anyway)."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise LLMError(f"No JSON object found in LLM output: {raw_text!r}")
    return json.loads(match.group(0))
