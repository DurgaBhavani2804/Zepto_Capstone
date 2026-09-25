"""
Task 6: FastAPI wrapper around the LangGraph pipeline.

Run locally (from inside support_assistant/):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 7860

Then:
    curl -X POST http://localhost:7860/ask \\
         -H "Content-Type: application/json" \\
         -d '{"query": "How long does delivery take?"}'

MOCK_LLM is left at its default (unset / "1") for the graded baseline — no
API key or network call to any LLM provider is required.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from graph import get_graph
from schemas import AskRequest, AskResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based support assistant over Zepto's policy corpus (LangGraph + ChromaDB + FastAPI).",
    version="1.0.0",
)


@app.on_event("startup")
def _startup():
    # Build/load the ChromaDB collection and compile the graph once at boot,
    # rather than on every request.
    get_graph()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    graph = get_graph()
    result = graph.invoke({"query": request.query})

    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"] or [],
        confidence=result["confidence"],
    )


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
