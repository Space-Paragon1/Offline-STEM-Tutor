from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None

@dataclass
class Chunk:
    text: str
    source: str
    page: int

class FaissStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.pkl")

        self.index = None
        self.meta: List[Chunk] = []

    def load(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available. Install faiss-cpu in this venv.")
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = None
            self.meta = []

    def save(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available.")
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # cosine if normalized
        self.index.add(embeddings)
        self.meta.extend(chunks)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or not self.meta:
            return []
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)

        scores, idx = self.index.search(query_emb, top_k)
        results = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.meta):
                continue
            c = self.meta[i]
            results.append({"score": float(score), "text": c.text, "source": c.source, "page": c.page})
        return results
