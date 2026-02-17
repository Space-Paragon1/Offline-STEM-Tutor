from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.retriever_vec import build_vec_index, load_vec_index, retrieve_vec

from tools_chem import molar_mass, calc_molarity
from rag.ingest import ingest_folder
from rag.retriever_bm25 import (
    load_chunks,
    build_bm25_index,
    load_bm25_index,
    retrieve,
)
from tutor.prompts import build_prompt

# -----------------------------
# Config / Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.gguf")

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_DIR = os.path.join(DATA_DIR, "index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
VEC_PATH = os.path.join(INDEX_DIR, "vec.pkl")

# -----------------------------
# Globals
# -----------------------------
llm = None
llm_error: Optional[str] = None

rag_chunks: List[Dict[str, Any]] = []
bm25 = None
vec_index = None


# -----------------------------
# Request/Response Types
# -----------------------------
class HistoryItem(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    subject: str = "cs"     # "cs" or "chem"
    mode: str = "explain"   # explain | socratic | hints | quiz
    history: List[HistoryItem] = []


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []


# -----------------------------
# LLM loader
# -----------------------------
def load_llm() -> None:
    """Loads llama.cpp model into memory (CPU)."""
    global llm, llm_error
    if llm is not None or llm_error is not None:
        return

    if not os.path.exists(MODEL_PATH):
        llm_error = f"Model file not found at: {MODEL_PATH}"
        return

    try:
        # Disable CPU_REPACK buffer (saves ~1.3 GB RAM in llama-cpp-python 0.3.x)
        os.environ.setdefault("GGML_CPU_REPACK", "0")

        from llama_cpp import Llama

        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=max(1, (os.cpu_count() or 4) // 2),
            n_batch=64,
            n_gpu_layers=0,
            verbose=False,
        )
    except Exception as e:
        llm_error = f"Failed to load model: {e!r}"


# -----------------------------
# Lifespan (replaces deprecated on_event)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_llm()

    global rag_chunks, bm25, vec_index
    rag_chunks = load_chunks(CHUNKS_PATH)
    bm25 = load_bm25_index(BM25_PATH)
    if os.path.exists(VEC_PATH):
        vec_index = load_vec_index(VEC_PATH)

    yield  # app runs here


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Offline STEM Tutor Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Formula extraction
# -----------------------------
def extract_formula(text: str) -> str | None:
    """
    Extract a chemical formula from a question.
    Valid formulas must start with an uppercase letter and:
      - contain at least one digit or parenthesis (e.g. H2O, Ca(OH)2), OR
      - be a 1-4 char all-alpha token starting uppercase (e.g. NaCl, Fe, KBr)
    """
    # Match tokens starting with uppercase
    matches = re.findall(r"\b([A-Z][A-Za-z0-9()]*)\b", text)
    if not matches:
        return None

    blacklist = {
        "what", "is", "the", "molar", "mass", "of", "in", "and", "for",
        "calculate", "find", "give", "with", "molarity", "concentration",
        "solution", "water", "acid", "base", "salt", "compound", "element",
        "reaction", "equation", "product", "reactant", "yield", "moles",
        "grams", "liters", "volume", "answer", "please", "tell", "how",
        "many", "much", "need", "want", "if", "then", "from", "into",
    }

    def score(m: str) -> tuple:
        has_formula_chars = any(c.isdigit() or c in "()" for c in m)
        return (0 if has_formula_chars else 1, -len(m))

    for m in sorted(matches, key=score):
        if m.lower() not in blacklist:
            return m

    return None


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "llm_loaded": llm is not None,
        "llm_error": llm_error,
        "docs_dir": DOCS_DIR,
        "chunks_path_exists": os.path.exists(CHUNKS_PATH),
        "bm25_path_exists": os.path.exists(BM25_PATH),
        "chunks_loaded": len(rag_chunks),
        "bm25_loaded": bm25 is not None,
        "vec_path_exists": os.path.exists(VEC_PATH),
        "vec_loaded": vec_index is not None,
    }


@app.post("/ingest")
def ingest():
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    ingest_folder(DOCS_DIR, CHUNKS_PATH)

    chunks = load_chunks(CHUNKS_PATH)
    build_bm25_index(chunks, BM25_PATH)

    global rag_chunks, bm25, vec_index
    rag_chunks = chunks
    bm25 = load_bm25_index(BM25_PATH)
    vec_index = build_vec_index(chunks, VEC_PATH)

    return {"ok": True, "chunks": len(chunks), "docs_dir": DOCS_DIR}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    load_llm()

    question = (req.question or "").strip()

    print("== /ask ==")
    print("subject:", req.subject)
    print("question:", question)

    if not question:
        return AskResponse(answer="Please type a question.", citations=[])

    # -----------------------------
    # FAST CHEM TOOLS (bypass LLM)
    # -----------------------------
    if req.subject.strip().lower() == "chem":
        q_lower = question.lower()

        # TOOL: molar mass
        if "molar mass" in q_lower:
            formula = extract_formula(question)
            if not formula:
                return AskResponse(
                    answer="Tell me the chemical formula (e.g., 'molar mass of H2SO4' or 'molar mass of Ca(OH)2').",
                    citations=[],
                )
            try:
                mm = molar_mass(formula)
                print("ROUTING -> TOOL: molar_mass")
                return AskResponse(
                    answer=f"Molar mass of **{formula}** ≈ **{mm:.3f} g/mol**.",
                    citations=[],
                )
            except Exception as e:
                return AskResponse(answer=f"I couldn't compute that molar mass: {e}", citations=[])

        # TOOL: molarity
        if "molarity" in q_lower or "molar concentration" in q_lower:
            try:
                mol_match = re.search(r"([-+]?\d*\.?\d+)\s*(mol|moles?)\b", q_lower)
                vol_match = re.search(r"([-+]?\d*\.?\d+)\s*(ml|l)\b", q_lower)

                if mol_match and vol_match:
                    moles = float(mol_match.group(1))
                    vol_val = float(vol_match.group(1))
                    vol_unit = vol_match.group(2).upper()

                    M = calc_molarity(moles=moles, volume_value=vol_val, volume_unit=vol_unit)
                    print("ROUTING -> TOOL: calc_molarity (moles/vol)")
                    return AskResponse(
                        answer=f"**Molarity** M = n/V = {moles} mol / {vol_val} {vol_unit} = **{M:.4f} M**",
                        citations=[],
                    )

                mass_match = re.search(r"([-+]?\d*\.?\d+)\s*(g|grams?)\b", q_lower)
                vol_match2 = re.search(r"([-+]?\d*\.?\d+)\s*(ml|l)\b", q_lower)
                formula = extract_formula(question)

                if mass_match and vol_match2 and formula:
                    mass_g = float(mass_match.group(1))
                    vol_val = float(vol_match2.group(1))
                    vol_unit = vol_match2.group(2).upper()

                    M = calc_molarity(mass_g=mass_g, formula=formula, volume_value=vol_val, volume_unit=vol_unit)
                    print("ROUTING -> TOOL: calc_molarity (grams/formula/vol)")
                    return AskResponse(
                        answer=f"**Molarity** ≈ **{M:.4f} M** (from {mass_g} g of {formula} in {vol_val} {vol_unit})",
                        citations=[],
                    )

                return AskResponse(
                    answer=(
                        "For molarity, tell me either:\n"
                        "- **moles and volume** (e.g., `0.5 mol in 250 mL`)\n\n"
                        "or\n\n"
                        "- **grams + formula + volume** (e.g., `10 g NaCl in 500 mL`)"
                    ),
                    citations=[],
                )

            except Exception as e:
                return AskResponse(answer=f"I couldn't compute molarity: {e}", citations=[])

    # -----------------------------
    # LLM not loaded guard
    # -----------------------------
    if llm is None:
        return AskResponse(
            answer=f"LLM not loaded. Check `/health` for details.\n\nError: `{llm_error}`",
            citations=[],
        )

    # -----------------------------
    # Hybrid RAG retrieval
    # -----------------------------
    contexts: List[str] = []
    cites_out: List[Dict[str, Any]] = []
    context_blocks: List[Dict[str, Any]] = []

    contexts_bm25, cites_bm25 = [], []
    contexts_vec, cites_vec = [], []

    if bm25 is not None and rag_chunks:
        try:
            contexts_bm25, cites_bm25 = retrieve(question, rag_chunks, bm25, top_k=4)
        except Exception as e:
            print(f"[WARN] BM25 retrieve() failed: {e}")

    if vec_index is not None and rag_chunks:
        try:
            contexts_vec, cites_vec = retrieve_vec(question, rag_chunks, vec_index, top_k=4)
        except Exception as e:
            print(f"[WARN] Vector retrieve_vec() failed: {e}")

    # Merge: BM25 first, then vectors; dedupe by chunk_id
    seen: set = set()
    for ctx, c in list(zip(contexts_bm25, cites_bm25)) + list(zip(contexts_vec, cites_vec)):
        cid = getattr(c, "chunk_id", None) or "unknown"
        if cid in seen:
            continue
        seen.add(cid)
        contexts.append(ctx)
        cite_dict = {
            "source": getattr(c, "source", None),
            "page": getattr(c, "page", None),
            "chunk_id": getattr(c, "chunk_id", None),
            "snippet": getattr(c, "snippet", None),
        }
        cites_out.append(cite_dict)
        context_blocks.append({
            "text": ctx[:800],
            "source": cite_dict["source"],
            "page": cite_dict["page"],
        })

    # Clamp to top 3 context blocks
    context_blocks = context_blocks[:3]
    cites_out = cites_out[:3]

    # -----------------------------
    # Build prompt via tutor/prompts.py
    # -----------------------------
    history = [{"role": h.role, "content": h.content} for h in req.history]

    prompt = build_prompt(
        subject=req.subject,
        question=question,
        mode=req.mode,
        context_blocks=context_blocks if context_blocks else None,
        history=history if history else None,
    )

    # -----------------------------
    # LLM generate
    # -----------------------------
    print("ROUTING -> LLM (slow path)")
    out = llm(
        prompt,
        max_tokens=350,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>", "\n\nStudent:", "\n\nUser:", "\n\nUSER:", "\n\nQuestion:", "</USER_QUESTION>"],
        echo=False,
    )

    text = out["choices"][0]["text"].strip()

    return AskResponse(answer=text, citations=cites_out)
