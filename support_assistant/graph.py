"""
Task 3 + 4: LangGraph StateGraph with a TypedDict state, 3 nodes, and a
conditional edge that routes between them based on classified intent.

Every node's generation step branches on the MOCK_LLM environment variable:
  - MOCK_LLM unset or "1" (default)  -> graded, fully offline, deterministic
    rule-based / templated logic. No LLM call, no network call.
  - MOCK_LLM="0"                     -> optional extension, calls a real LLM
    (see llm_client.py), with schema-validation retry logic.
"""

import os
from typing import List, Optional, TypedDict

from pydantic import ValidationError

from langgraph.graph import END, StateGraph

from ingest import get_or_build_collection
from llm_client import LLMError, call_llm, extract_json
from prompt_template import build_direct_prompt, build_rag_prompt
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

MAX_LLM_RETRIES = 2  # "retry up to 2 additional times" per the spec


def is_mock_mode() -> bool:
    """MOCK_LLM unset or '1' => mock (graded baseline). MOCK_LLM='0' => real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: Optional[str]  # "policy_question" | "general_question"
    retrieved_chunks: Optional[List[dict]]  # [{"id": str, "document": str, "distance": float}]
    answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = (
            "policy_question"
            if any(kw in lowered for kw in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        # Optional MOCK_LLM=0 extension: ask the LLM to classify.
        classify_prompt = (
            "Classify the following customer question as exactly one word: "
            "either 'policy_question' (about Zepto delivery, returns, refunds, "
            "membership, tracking, cancellation, gift cards, or support hours) "
            "or 'general_question' (anything else). "
            f"Question: {query}\nAnswer with only the single classification word."
        )
        try:
            raw = call_llm(classify_prompt)
            intent = "policy_question" if "policy_question" in raw.lower() else "general_question"
        except LLMError:
            # Fail safe: fall back to the keyword heuristic if the LLM call fails.
            lowered = query.lower()
            intent = (
                "policy_question"
                if any(kw in lowered for kw in POLICY_KEYWORDS)
                else "general_question"
            )

    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Routing function for the conditional edge out of classify_intent
# ---------------------------------------------------------------------------
def route_after_classify(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer  (policy_question path)
# ---------------------------------------------------------------------------
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_or_build_collection()
    return _collection


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes (no API key needed).
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_chunks = [
        {"id": doc_id, "document": doc, "distance": dist}
        for doc_id, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]

    if is_mock_mode():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk = retrieved_chunks[0]["document"]
        top_chunk_snippet = top_chunk[:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        sources = [c["id"] for c in retrieved_chunks]
        confidence = 1.0
        return {
            **state,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    # Optional MOCK_LLM=0 extension: real LLM grounded in retrieved chunks,
    # with schema-validation retry logic.
    prompt = build_rag_prompt(query, retrieved_chunks)
    parsed = _call_llm_with_schema_retries(prompt)
    if parsed is None:
        return {
            **state,
            "retrieved_chunks": retrieved_chunks,
            "answer": None,
            "sources": None,
            "confidence": None,
            "error": "LLM output failed schema validation after retries.",
        }

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "answer": parsed.answer,
        "sources": parsed.sources,
        "confidence": parsed.confidence,
    }


# ---------------------------------------------------------------------------
# Node 3: direct_answer  (general_question path)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        return {
            **state,
            "retrieved_chunks": [],
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: real LLM, no retrieval.
    prompt = build_direct_prompt(query)
    parsed = _call_llm_with_schema_retries(prompt)
    if parsed is None:
        return {
            **state,
            "retrieved_chunks": [],
            "answer": None,
            "sources": None,
            "confidence": None,
            "error": "LLM output failed schema validation after retries.",
        }

    return {
        **state,
        "retrieved_chunks": [],
        "answer": parsed.answer,
        "sources": parsed.sources,
        "confidence": parsed.confidence,
    }


# ---------------------------------------------------------------------------
# Shared helper: call the LLM and validate/retry against AskResponse schema.
# Only ever invoked on the optional MOCK_LLM=0 path.
# ---------------------------------------------------------------------------
def _call_llm_with_schema_retries(prompt: str) -> Optional[AskResponse]:
    current_prompt = prompt
    last_error = None

    for attempt in range(MAX_LLM_RETRIES + 1):  # 1 initial try + 2 retries
        try:
            raw = call_llm(current_prompt)
            data = extract_json(raw)
            return AskResponse(**data)
        except (LLMError, ValidationError, ValueError, KeyError) as e:
            last_error = e
            current_prompt = (
                prompt
                + "\n\nYour previous response was invalid "
                + f"({e}). Respond again with ONLY a single valid JSON object "
                "matching the required schema exactly: "
                '{"answer": "<string>", "sources": ["<id>", ...], "confidence": <float 0-1>}'
            )

    # All attempts exhausted.
    print(f"[graph] LLM output failed schema validation after retries: {last_error}")
    return None


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
