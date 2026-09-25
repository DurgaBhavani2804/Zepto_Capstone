"""
Ingestion + retrieval layer.

Stage: ingestion -> embedding -> retrieval (the first three stages of the RAG
pipeline described in README.md).

- load_documents(): reads the 8 .txt files in docs/
- chunk_documents(): turns each document into one or more chunks (one chunk
  per document here, since each policy doc is already short and self-contained)
- get_embedding_model(): loads the sentence-transformers all-MiniLM-L6-v2 model
  once and reuses it (this always runs locally, regardless of MOCK_LLM - the
  toggle only affects the final answer-generation step, never embedding)
- get_collection(): returns a persistent ChromaDB collection, populating it
  from the corpus the first time it is empty
- retrieve(query, k): embeds the query with the same model and returns the
  top-k most similar chunks
"""

import os
from pathlib import Path
from typing import List, Tuple

import chromadb

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None          # lazy singleton for the sentence-transformers model
_client = None
_collection = None


def get_embedding_model():
    """Load (once) the local sentence-transformers embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(list(texts), convert_to_numpy=True).tolist()


def load_documents() -> List[Tuple[str, str]]:
    """Return [(doc_id, text), ...] for every docs/doc_XX.txt file, sorted."""
    docs = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem  # e.g. "doc_01"
        text = path.read_text(encoding="utf-8").strip()
        docs.append((doc_id, text))
    return docs


def chunk_documents(docs: List[Tuple[str, str]], max_chars: int = 800) -> List[Tuple[str, str]]:
    """
    Chunk each document. Each policy doc here is short and already a single
    coherent unit, so a document that fits under max_chars becomes exactly
    one chunk; a longer document (not the case in this corpus, but handled
    for robustness) is split into fixed-size chunks.
    """
    chunks = []
    for doc_id, text in docs:
        if len(text) <= max_chars:
            chunks.append((doc_id, text))
        else:
            for i in range(0, len(text), max_chars):
                piece = text[i:i + max_chars]
                chunks.append((f"{doc_id}_chunk{i // max_chars}", piece))
    return chunks


def get_collection():
    """Return the ChromaDB collection, ingesting the corpus if it's empty."""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(COLLECTION_NAME)

    if _collection.count() == 0:
        docs = load_documents()
        chunks = chunk_documents(docs)
        ids = [cid for cid, _ in chunks]
        texts = [text for _, text in chunks]
        embeddings = embed_texts(texts)
        _collection.add(ids=ids, documents=texts, embeddings=embeddings)

    return _collection


def retrieve(query: str, k: int = 3) -> List[Tuple[str, str]]:
    """
    Embed the query with the same sentence-transformers model and return the
    top-k most similar (chunk_id, chunk_text) pairs from ChromaDB, most
    similar first. This step always runs for real in both MOCK_LLM modes.
    """
    collection = get_collection()
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    ids = results["ids"][0]
    docs = results["documents"][0]
    return list(zip(ids, docs))
