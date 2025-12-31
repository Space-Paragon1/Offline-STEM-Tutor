from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    source: str          # filename
    page: int            # 1-based page number (0 if not applicable)
    text: str


_WORD_RE = re.compile(r"\w+")


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def read_pdf(path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append({"page": i, "text": normalize(txt)})
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking with overlap.
    chunk_size ~ 900 chars is fine for BM25.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_folder(docs_dir: str, out_chunks_path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for fname in os.listdir(docs_dir):
        path = os.path.join(docs_dir, fname)
        if not os.path.isfile(path):
            continue

        lower = fname.lower()
        if lower.endswith(".pdf"):
            pages = read_pdf(path)
            for p in pages:
                for j, ch in enumerate(chunk_text(p["text"])):
                    cid = f"{fname}::p{p['page']}::c{j}"
                    chunks.append(Chunk(id=cid, source=fname, page=p["page"], text=ch))

        elif lower.endswith(".txt") or lower.endswith(".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = normalize(f.read())
            for j, ch in enumerate(chunk_text(txt)):
                cid = f"{fname}::p0::c{j}"
                chunks.append(Chunk(id=cid, source=fname, page=0, text=ch))

    # save JSONL
    os.makedirs(os.path.dirname(out_chunks_path), exist_ok=True)
    with open(out_chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    return chunks
