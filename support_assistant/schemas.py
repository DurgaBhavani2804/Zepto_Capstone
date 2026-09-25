"""
Pydantic models for the /ask endpoint.

AskRequest  -> what the client sends.
AskResponse -> the schema every answer (mock or real-LLM) is validated against,
               per Task 5 of the assignment (answer / sources / confidence).
"""

from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated (or canned) answer text.")
    sources: List[str] = Field(
        default_factory=list,
        description="Chunk/document ids used to ground the answer. Empty for general_question.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1."
    )


class AskResponseError(BaseModel):
    """Returned only on the optional MOCK_LLM=0 path if the LLM output still
    fails schema validation after the retry budget is exhausted."""

    error: str
    detail: str
