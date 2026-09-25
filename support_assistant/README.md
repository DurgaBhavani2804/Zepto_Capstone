# Support Assistant (Module 3)

A small RAG service for Zepto: 8 policy documents, embedded locally and
retrieved with ChromaDB, orchestrated through a LangGraph intent router, with
a validated JSON output and a FastAPI wrapper.

Every LLM call in this module is gated behind the `MOCK_LLM` environment
variable. Left unset (the default, and what gets graded), the service is
fully deterministic and offline: no signup, no API key, no network call to
any LLM provider. Setting `MOCK_LLM=0` switches to an optional, ungraded
real-LLM extension using Groq's free tier.

## Files

| File | Purpose |
|---|---|
| `docs/doc_01.txt` … `doc_08.txt` | The 8-document policy corpus |
| `rag.py` | Ingestion, chunking, embedding (sentence-transformers), ChromaDB storage + retrieval |
| `prompts.py` | The structured prompt template (role/context/task/format/length, negative constraint, few-shot example) — used only by the optional `MOCK_LLM=0` path |
| `schemas.py` | Pydantic request/response models + the LangGraph `TypedDict` state |
| `graph.py` | The LangGraph `StateGraph`: 3 nodes, conditional routing edge, `MOCK_LLM` branch in each node |
| `llm.py` | Optional real-LLM call (Groq) + schema-validation retry logic, only used when `MOCK_LLM=0` |
| `main.py` | FastAPI app, `POST /ask` and `GET /health` |
| `Dockerfile` | Builds and runs the FastAPI app in a container |

## Architecture: how a request flows through the pipeline

**1. Ingestion** (`rag.py: load_documents`, `chunk_documents`) — On the first
app startup, the 8 files in `docs/` are read from disk. Each policy document
is short and self-contained, so the simple, task-permitted chunking scheme
used here is one chunk per document (8 chunks total, ids `doc_01` … `doc_08`).
A fixed-size fallback (`max_chars`-based splitting) is included in the same
function for robustness, in case a document were ever long enough to need it.

**2. Embedding** (`rag.py: get_embedding_model`, `embed_texts`) — Each chunk's
text is embedded locally with `sentence-transformers/all-MiniLM-L6-v2`
(no API key, no account, runs on CPU). The resulting vectors are stored in a
persistent ChromaDB collection called `zepto_policies`
(`rag.py: get_collection`), so ingestion only happens once — later app starts
reuse the collection already saved to disk in `chroma_db/`. This stage
**never** depends on `MOCK_LLM`; it always runs for real in both modes.

**3. Retrieval** (`rag.py: retrieve`, called from `graph.py: retrieve_and_answer`)
— When a query needs policy context, it is embedded with the same MiniLM
model, and ChromaDB returns the top-3 most similar chunks by cosine
similarity. Like embedding, retrieval always runs for real in both modes.

**4. Routing** (`graph.py: classify_intent`, `_route`) — The LangGraph entry
node classifies the query as `policy_question` or `general_question`. In the
default mock mode this is a plain keyword heuristic (checks for words like
"delivery", "return", "refund", …) with no LLM call at all. A conditional
edge then sends the state to either `retrieve_and_answer` or `direct_answer`.
This routing decision itself never depends on `MOCK_LLM` — only what happens
*inside* the two downstream nodes does.

**5. Generation** (`graph.py: retrieve_and_answer` / `direct_answer`) — This
is the only stage that branches on `MOCK_LLM`:
- **Default (`MOCK_LLM` unset or `1`, graded baseline):** no LLM call at all.
  `retrieve_and_answer` returns the canned template
  `f"Based on the retrieved context: {top_chunk_snippet}"` using the top
  retrieved chunk; `direct_answer` returns the fixed string *"I can only
  answer questions about Zepto policies right now."* The output schema
  (`answer` / `sources` / `confidence`) is populated directly by this code,
  deterministically, since there is no LLM output to validate.
- **Optional (`MOCK_LLM=0`):** `retrieve_and_answer` fills the structured
  prompt in `prompts.py` with the retrieved chunks and asks a real LLM
  (Groq, `llm.py`) to answer grounded only in that context; `direct_answer`
  prompts the LLM with no retrieved context. In both cases the raw LLM
  output is parsed as JSON and validated against the `AskResponse` Pydantic
  model; on a validation failure it retries up to 2 more times with a
  corrective instruction (`llm.py: generate_structured_answer`) before
  falling back to a clearly marked `[error] ...` response.

**6. API** (`main.py`) — `POST /ask` takes `{"query": str}`, invokes the
compiled LangGraph (`graph.py: app_graph`), and returns the validated
`AskResponse` JSON.

```
docs/*.txt --(load+chunk)--> rag.chunk_documents --(embed: MiniLM)-->
ChromaDB(zepto_policies) <--(retrieve top-3)-- graph.retrieve_and_answer
                                                        |
query --> graph.classify_intent --(conditional edge)--> graph.retrieve_and_answer --> AskResponse
                                \-----------------------> graph.direct_answer -------> AskResponse
```

## Running it locally

```bash
cd support_assistant
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

The first request triggers ingestion (downloads the MiniLM model weights the
very first time only, then reuses them; embeds the 8 docs into `chroma_db/`).

## Running it in Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

## Example calls (MOCK_LLM left at its default)


**1. A query that should trigger retrieval** (contains a policy keyword):
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does delivery take and is it free?"}'
```
Response:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1
}
```

**2. A query that should NOT trigger retrieval** (no policy keyword):
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```
Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1
}
```

## Optional extensions (not required for full marks)

- **Real LLM:** set `MOCK_LLM=0` and `GROQ_API_KEY=<your free-tier key>`
  (from https://console.groq.com) before starting the server.

