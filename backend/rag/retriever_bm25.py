from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi

from .ingest import tokenize


@dataclass
class Citation:
    source: str
    page: int
    chunk_id: str
    snippet: str


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.exists(chunks_path):
        return chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_bm25_index(chunks: List[Dict[str, Any]], out_index_path: str) -> None:
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    os.makedirs(os.path.dirname(out_index_path), exist_ok=True)
    with open(out_index_path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25_index(index_path: str):
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve(
    query: str,
    chunks: List[Dict[str, Any]],
    bm25,
    top_k: int = 4,
) -> Tuple[List[str], List[Citation]]:
    if not chunks or bm25 is None:
        return [], []

    qtok = tokenize(query)
    scores = bm25.get_scores(qtok)

    # get top_k indices
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    contexts = []
    citations: List[Citation] = []

    for idx in ranked:
        c = chunks[idx]
        txt = c["text"].strip()
        contexts.append(f"[{c['id']}] ({c['source']} p.{c['page']})\n{txt}")
        citations.append(
            Citation(
                source=c["source"],
                page=int(c["page"]),
                chunk_id=c["id"],
                snippet=txt[:240].replace("\n", " "),
            )
        )

    return contexts, citations
