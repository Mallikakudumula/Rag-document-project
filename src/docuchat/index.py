"""Ingest = load -> chunk -> (optional embed) -> save to a JSON index file.
In production you'd swap this JSON file for a vector DB (Azure AI Search, pgvector, Qdrant)."""
import json
import logging
from pathlib import Path

from .chunking import Chunk, build_chunks
from .config import Settings
from .loaders import load_folder
from .retrieval import Retriever, SentenceTransformerEmbedder

log = logging.getLogger(__name__)


def build_index(s: Settings) -> int:
    chunks = build_chunks(load_folder(s.docs_dir), s.chunk_words, s.chunk_overlap)
    if not chunks:
        raise SystemExit(f"No supported documents found in {s.docs_dir}")
    data = {"chunks": [c.to_dict() for c in chunks]}
    if s.use_embeddings:
        emb = SentenceTransformerEmbedder().embed([c.text for c in chunks])
        data["embeddings"] = emb.tolist()
    s.index_path.parent.mkdir(parents=True, exist_ok=True)
    s.index_path.write_text(json.dumps(data))
    log.info("Indexed %d chunks from %s", len(chunks), s.docs_dir)
    return len(chunks)


def load_retriever(s: Settings) -> Retriever:
    if not Path(s.index_path).exists():
        build_index(s)
    data = json.loads(Path(s.index_path).read_text())
    chunks = [Chunk(**c) for c in data["chunks"]]
    if "embeddings" in data and s.use_embeddings:
        import numpy as np
        return Retriever(chunks, SentenceTransformerEmbedder(), np.array(data["embeddings"]))
    return Retriever(chunks)
