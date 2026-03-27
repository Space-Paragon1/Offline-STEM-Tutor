from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Cite:
    source: str | None
    page: int | None
    chunk_id: str | None
    snippet: str | None

def _chunk_text(chunk: Dict[str, Any]) -> str:
    return (chunk.get("text") or "").strip()

def build_vec_index(chunks: List[Dict[str, Any]], out_path: str, model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Builds a vector index and saves it as a pickle:
    {
      "model_name": str,
      "embeddings": np.ndarray [N, D],
    }
    """
    model = SentenceTransformer(model_name)
    texts = [_chunk_text(c) for c in chunks]

    # Encode in batches to reduce RAM spikes
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"model_name": model_name, "embeddings": embs}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return payload

def load_vec_index(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def retrieve_vec(
    query: str,
    chunks: List[Dict[str, Any]],
    vec_index: Dict[str, Any],
    top_k: int = 4,
) -> Tuple[List[str], List[Cite]]:
    """
    Returns (contexts, citations) using cosine similarity on normalized embeddings.
    """
    if not vec_index or "embeddings" not in vec_index:
        return [], []

    model_name = vec_index.get("model_name", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    embs = vec_index["embeddings"]

    # cosine similarity since embeddings are normalized
    scores = embs @ q_emb
    idxs = np.argsort(-scores)[:top_k]

    contexts: List[str] = []
    cites: List[Cite] = []

    for i in idxs:
        ch = chunks[int(i)]
        text = (ch.get("text") or "").strip()
        contexts.append(text)

        cites.append(
            Cite(
                source=ch.get("source"),
                page=ch.get("page"),
                chunk_id=str(ch.get("chunk_id") or i),
                snippet=text[:220],
            )
        )

    return contexts, cites
