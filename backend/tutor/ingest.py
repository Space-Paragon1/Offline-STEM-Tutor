from __future__ import annotations
import os, re
from typing import List, Tuple
from pypdf import PdfReader
from .retriever import Chunk

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, source: str, page: int, chunk_size: int = 800, overlap: int = 120) -> List[Chunk]:
    text = _clean(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(Chunk(text=chunk, source=source, page=page))
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def ingest_pdf(pdf_path: str) -> List[Chunk]:
    reader = PdfReader(pdf_path)
    out: List[Chunk] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        out.extend(chunk_text(txt, source=os.path.basename(pdf_path), page=i + 1))
    return out

def ingest_txt(txt_path: str) -> List[Chunk]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return chunk_text(txt, source=os.path.basename(txt_path), page=1)
