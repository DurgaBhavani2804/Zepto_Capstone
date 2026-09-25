"""
FastAPI wrapper around the LangGraph support-assistant pipeline.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

Then:
    POST http://localhost:7860/ask   {"query": "How long does delivery take?"}
"""

from fastapi import FastAPI

from schemas import AskRequest, AskResponse
from graph import app_graph
from rag import get_collection

app = FastAPI(title="Zepto Support Assistant")


@app.on_event("startup")
def startup() -> None:
    # Ingest the 8-document corpus into ChromaDB the first time the app
    # starts (a no-op on later starts, since the collection persists to disk).
    get_collection()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = app_graph.invoke({"query": request.query})
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0),
    )
