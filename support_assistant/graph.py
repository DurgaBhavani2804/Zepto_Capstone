"""
LangGraph orchestration.

Nodes:
  classify_intent      - routes each query to policy_question / general_question
  retrieve_and_answer  - retrieval (always real) + answer generation (branches on MOCK_LLM)
  direct_answer        - answer generation with no retrieval (branches on MOCK_LLM)

MOCK_LLM (env var) gates every generation step:
  unset or "1" (default, graded baseline) -> deterministic, rule-based, no network call
  "0" (optional extension)                -> calls a real LLM (see llm.py)
"""

import os

from langgraph.graph import StateGraph, END

from schemas import GraphState
from rag import retrieve

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]


def _mock_mode() -> bool:
    """True unless MOCK_LLM is explicitly set to '0'."""
    return os.environ.get("MOCK_LLM", "1") != "0"


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if _mock_mode():
        # Mock / graded baseline: keyword heuristic, no LLM call.
        q = query.lower()
        intent = "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: ask the LLM to classify instead.
        from llm import _call_groq
        raw = _call_groq(
            "Classify this user question as exactly one word, either "
            "'policy_question' (about Zepto delivery/returns/membership/"
            "tracking/cancellation/gift cards/support hours) or "
            f"'general_question' (anything else). Question: {query!r}\n"
            "Reply with exactly one of those two words."
        ).strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"

    state["intent"] = intent
    return state


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real, in both modes - embeddings and ChromaDB
    # need no API key and no network call to any LLM provider.
    hits = retrieve(query, k=3)              # [(chunk_id, chunk_text), ...] best-first
    ids = [cid for cid, _ in hits]

    if _mock_mode():
        # Mock / graded baseline: canned templated answer, no LLM call.
        top_chunk_snippet = hits[0][1][:200] if hits else ""
        state["answer"] = f"Based on the retrieved context: {top_chunk_snippet}"
        state["sources"] = ids
        state["confidence"] = 1.0
    else:
        # Optional MOCK_LLM=0 extension: real LLM grounded on retrieved chunks,
        # with schema-validation retry logic (see llm.py).
        from llm import generate_structured_answer
        from prompts import build_answer_prompt
        prompt = build_answer_prompt(query, hits)
        result = generate_structured_answer(prompt)
        state["answer"] = result.answer
        state["sources"] = result.sources or ids
        state["confidence"] = result.confidence

    return state


def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if _mock_mode():
        # Mock / graded baseline: fixed canned string, no LLM call, no retrieval.
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        from llm import generate_structured_answer
        from prompts import build_direct_prompt
        prompt = build_direct_prompt(query)
        result = generate_structured_answer(prompt)
        state["answer"] = result.answer
        state["sources"] = []
        state["confidence"] = result.confidence

    return state


def _route(state: GraphState) -> str:
    """Conditional-edge function: reads the intent set by classify_intent."""
    return state["intent"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


app_graph = build_graph()
