"""
Task 1: Load the 8 corpus documents, chunk them, embed each chunk locally
with sentence-transformers/all-MiniLM-L6-v2, and store the embeddings in a
ChromaDB collection.

No API key and no network call to any LLM provider is needed for this step —
sentence-transformers and ChromaDB both run entirely on your machine (the
embedding model itself is downloaded once from Hugging Face the first time
you run this, which requires normal internet access but no account/API key).

Chunking scheme: each policy document here is short (a single paragraph), so
we use a simple per-document chunk — one chunk == one document. This is the
"simple per-document chunk" option explicitly allowed by the assignment.
"""

import glob
import os

import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Load every doc_*.txt file and turn it into one chunk each.

    Returns a list of {"id": "doc_01", "document": "<text>"} dicts.
    """
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "document": text})
    return chunks


def get_or_build_collection(
    persist_directory: str = CHROMA_DIR,
    docs_dir: str = DOCS_DIR,
    force_rebuild: bool = False,
):
    """Return a ChromaDB collection with all 8 chunks embedded and stored.

    Uses ChromaDB's built-in SentenceTransformerEmbeddingFunction so both
    ingestion-time and query-time embeddings use the exact same model
    (all-MiniLM-L6-v2), which is what makes cosine-similarity retrieval
    meaningful.
    """
    client = chromadb.PersistentClient(path=persist_directory)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Only (re)embed if the collection is empty, so repeated app restarts
    # don't re-embed on every boot.
    if collection.count() == 0:
        chunks = load_documents(docs_dir)
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["document"] for c in chunks],
            metadatas=[{"source": c["id"]} for c in chunks],
        )

    return collection


if __name__ == "__main__":
    col = get_or_build_collection(force_rebuild=True)
    print(f"Collection '{COLLECTION_NAME}' now has {col.count()} chunks.")
    res = col.query(query_texts=["How long does delivery take?"], n_results=3)
    for doc_id, doc, dist in zip(
        res["ids"][0], res["documents"][0], res["distances"][0]
    ):
        print(f"- {doc_id} (distance={dist:.4f}): {doc[:80]}...")
