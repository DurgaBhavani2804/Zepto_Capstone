"""Pydantic models (API contract) and the LangGraph TypedDict state."""

from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float


class GraphState(TypedDict, total=False):
    """State object threaded through the LangGraph nodes."""
    query: str
    intent: str                # "policy_question" | "general_question"
    retrieved_ids: List[str]
    retrieved_docs: List[str]
    answer: str
    sources: List[str]
    confidence: float
