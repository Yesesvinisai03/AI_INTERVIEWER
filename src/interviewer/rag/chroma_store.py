from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------
# Config (SAFE LIMITS)
# --------------------------------------------------
DEFAULT_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_RESUME_CHARS = 30_000     # hard cap (~6 pages)
MAX_CHUNKS = 40               # prevents memory blowups
CHUNK_SIZE = 800
OVERLAP = 100


# --------------------------------------------------
# Utils
# --------------------------------------------------
def _hash_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="ignore")
    ).hexdigest()[:16]

# Clean extracted resume text by removing noise
# Filters out headers, footers, and very short lines
def clean_resume_text(text: str) -> str:
    """
    Remove garbage lines commonly found in PDFs
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        if line.lower().startswith(("page ", "copyright", "resume")):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)

# Split resume text into overlapping chunks
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    max_chunks: int = MAX_CHUNKS
) -> List[str]:

    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text) and len(chunks) < max_chunks:
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# --------------------------------------------------
# Resume RAG Store
# --------------------------------------------------
# Wrapper class for resume-based Retrieval Augmented Generation (RAG)
# Handles embedding, storage, and retrieval of resume content

@dataclass
class ResumeRAG:
    persist_dir: str = DEFAULT_PERSIST_DIR

    def __post_init__(self):
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir
        )

        # ✅ Stable, torch-backed embedding
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

    # Initialize ChromaDB client and embedding function
    def collection_name(self, user_id: str) -> str:
        safe = "".join(
            c for c in user_id if c.isalnum() or c in ("_", "-")
        )[:40]
        return f"resume_{safe or 'default'}"

    # Generate a safe, user-specific collection name
    def get_collection(self, user_id: str):
        return self.client.get_or_create_collection(
            name=self.collection_name(user_id),
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # Index resume text into the vector database
    # Applies cleaning, chunking, hashing, and caching
    def index_resume(
        self,
        user_id: str,
        resume_text: str
    ) -> Dict[str, Any]:

        resume_text = (resume_text or "").strip()
        if not resume_text:
            return {"ok": False, "reason": "empty resume"}

        # 🔒 HARD LIMIT SIZE
        if len(resume_text) > MAX_RESUME_CHARS:
            resume_text = resume_text[:MAX_RESUME_CHARS]

        resume_text = clean_resume_text(resume_text)

        chunks = chunk_text(resume_text)
        if not chunks:
            return {"ok": False, "reason": "no valid chunks"}

        col = self.get_collection(user_id)

        resume_hash = _hash_text(resume_text)

        # Avoid re-indexing same resume
        existing = col.get(where={"resume_hash": resume_hash})

        if existing and existing.get("ids"):
            return {
                "ok": True,
                "cached": True,
                "chunks": len(existing["ids"])
            }

        ids = [
            f"{resume_hash}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "chunk_id": i,
                "resume_hash": resume_hash
            }
            for i in range(len(chunks))
        ]

        col.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        return {
            "ok": True,
            "cached": False,
            "chunks": len(chunks)
        }

    # Used during question generation for RAG
    def retrieve(
        self,
        user_id: str,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:

        if not query.strip():
            return []

        col = self.get_collection(user_id)

        res = col.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        return [
            {
                "text": doc,
                "distance": float(dist),
                "meta": meta
            }
            for doc, dist, meta in zip(docs, dists, metas)
        ]
