from __future__ import annotations

import os
import re
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
# App
# -----------------------------
app = FastAPI(title="Offline STEM Tutor Backend")

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
# Request/Response Types
# -----------------------------
class AskRequest(BaseModel):
    question: str
    subject: str = "cs"     # "cs" or "chem"
    mode: str = "explain"   # explain | socratic | hints | quiz


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []


# -----------------------------
# Globals
# -----------------------------
llm = None
llm_error: Optional[str] = None

rag_chunks: List[Dict[str, Any]] = []
bm25 = None
vec_index = None



SYSTEM_PROMPT = """You are an offline STEM tutor.
Rules:
- Be clear and step-by-step.
- If the user asks for practice, give 3 problems and answers at the end.
- For Chemistry: show units, significant steps, and short explanations.
- For CS: explain concepts, give small examples, and warn about common mistakes.
- If you are unsure, say so and ask a brief follow-up question.
"""


def mode_instructions(mode: str) -> str:
    m = (mode or "explain").strip().lower()
    if m not in ("explain", "socratic", "hints", "quiz"):
        m = "explain"

    if m == "explain":
        return (
            "Mode: EXPLAIN\n"
            "- Give a clear, step-by-step explanation.\n"
            "- Use small examples.\n"
            "- End with a 2-line summary.\n"
        )

    if m == "socratic":
        return (
            "Mode: SOCRATIC\n"
            "- Ask 1–3 guiding questions first.\n"
            "- Do NOT give the final answer immediately.\n"
            "- If the user insists, give the final answer.\n"
        )

    if m == "hints":
        return (
            "Mode: HINTS\n"
            "- Give exactly ONE hint.\n"
            "- Do not give the full solution unless asked.\n"
            "- End by asking: 'Want another hint or the full solution?'\n"
        )

    return (
        "Mode: QUIZ\n"
        "- Create 3 practice questions.\n"
        "- Do NOT include answers unless the user asks.\n"
    )


def extract_formula(text: str) -> str | None:
    """
    Extract a likely chemical formula from a question.
    Handles parentheses and strips punctuation.
    Examples: Ca(OH)2, H2SO4, NaCl
    """
    matches = re.findall(r"[A-Za-z][A-Za-z0-9()]*", text)
    if not matches:
        return None

    matches.sort(key=len, reverse=True)

    blacklist = {"what", "is", "the", "molar", "mass", "of", "in", "and", "for"}
    for m in matches:
        if m.lower() not in blacklist and any(ch.isalpha() for ch in m):
            return m

    return None


def load_llm() -> None:
    """Loads llama.cpp model into memory (CPU)."""
    global llm, llm_error
    if llm is not None or llm_error is not None:
        return

    if not os.path.exists(MODEL_PATH):
        llm_error = f"Model file not found at: {MODEL_PATH}"
        return

    try:
        from llama_cpp import Llama

        # Safer defaults for Windows i5 / 8GB RAM
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=1024,  # smaller is faster + avoids stalls
            n_threads=max(1, (os.cpu_count() or 4) // 2),
            n_batch=128,
            verbose=False,
        )
    except Exception as e:
        llm_error = f"Failed to load model: {e!r}"


@app.on_event("startup")
def _startup() -> None:
    load_llm()

    global rag_chunks, bm25, vec_index

    # Load BM25 chunks/index if they exist
    rag_chunks = load_chunks(CHUNKS_PATH)
    bm25 = load_bm25_index(BM25_PATH)

    # Load vector (embedding) index if it exists
    if os.path.exists(VEC_PATH):
        vec_index = load_vec_index(VEC_PATH)



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

    # 1) Create chunks.jsonl from PDFs/TXT/MD in DOCS_DIR
    ingest_folder(DOCS_DIR, CHUNKS_PATH)

    # 2) Load chunks and build BM25 index
    chunks = load_chunks(CHUNKS_PATH)
    build_bm25_index(chunks, BM25_PATH)

    # 3) Put them into memory so /ask can retrieve immediately
    global rag_chunks, bm25
    rag_chunks = chunks
    bm25 = load_bm25_index(BM25_PATH)
    global vec_index
    vec_index = build_vec_index(chunks, VEC_PATH)

    return {"ok": True, "chunks": len(chunks), "docs_dir": DOCS_DIR}


def build_prompt(subject: str, question: str, mode: str) -> str:
    subject = (subject or "cs").strip().lower()
    if subject not in ("cs", "chem"):
        subject = "cs"

    return (
        f"{SYSTEM_PROMPT}\n"
        f"{mode_instructions(mode)}\n"
        f"Subject: {subject}\n"
        f"User question: {question.strip()}\n"
        f"Answer:\n"
        f"(Stop after finishing your answer.)\n"
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    load_llm()

    question = (req.question or "").strip()

    # ---- DEBUG PRINTS ----
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
                    answer=f"Molar mass of {formula} ≈ {mm:.3f} g/mol.",
                    citations=[],
                )
            except Exception as e:
                return AskResponse(answer=f"I couldn't compute that molar mass: {e}", citations=[])

        # TOOL: molarity
        if "molarity" in q_lower or "molar concentration" in q_lower:
            try:
                # Very simple extraction rules:
                # Look for patterns like: "<moles> mol" and "<volume> mL/L"
                mol_match = re.search(r"([-+]?\d*\.?\d+)\s*(mol|moles?)\b", q_lower)
                vol_match = re.search(r"([-+]?\d*\.?\d+)\s*(ml|l)\b", q_lower)

                if mol_match and vol_match:
                    moles = float(mol_match.group(1))
                    vol_val = float(vol_match.group(1))
                    vol_unit = vol_match.group(2).upper()

                    M = calc_molarity(moles=moles, volume_value=vol_val, volume_unit=vol_unit)
                    print("ROUTING -> TOOL: calc_molarity (moles/vol)")
                    return AskResponse(
                        answer=f"Molarity M = n/V = {moles} mol / {vol_val} {vol_unit} = {M:.4f} M",
                        citations=[],
                    )

                # Alternative: "molarity of 10 g NaCl in 500 mL"
                mass_match = re.search(r"([-+]?\d*\.?\d+)\s*(g|grams?)\b", q_lower)
                vol_match = re.search(r"([-+]?\d*\.?\d+)\s*(ml|l)\b", q_lower)
                formula = extract_formula(question)

                if mass_match and vol_match and formula:
                    mass_g = float(mass_match.group(1))
                    vol_val = float(vol_match.group(1))
                    vol_unit = vol_match.group(2).upper()

                    M = calc_molarity(mass_g=mass_g, formula=formula, volume_value=vol_val, volume_unit=vol_unit)
                    print("ROUTING -> TOOL: calc_molarity (grams/formula/vol)")
                    return AskResponse(
                        answer=f"Molarity ≈ {M:.4f} M (from {mass_g} g of {formula} in {vol_val} {vol_unit})",
                        citations=[],
                    )

                return AskResponse(
                    answer="For molarity, tell me either:\n"
                           "- moles and volume (e.g., '0.5 mol in 250 mL')\n"
                           "OR\n"
                           "- grams + formula + volume (e.g., '10 g NaCl in 500 mL')",
                    citations=[],
                )

            except Exception as e:
                return AskResponse(answer=f"I couldn't compute molarity: {e}", citations=[])

    # -----------------------------
    # If LLM isn't loaded
    # -----------------------------
    if llm is None:
        return AskResponse(
            answer=f"LLM not loaded. Check /health for details. Error: {llm_error}",
            citations=[],
        )

        # -----------------------------
    # Retrieval (Hybrid RAG: BM25 + Vector)
    # -----------------------------
    contexts: List[str] = []
    cites_out: List[Dict[str, Any]] = []

    contexts_bm25, cites_bm25 = ([], [])
    contexts_vec, cites_vec = ([], [])

    # 1) BM25 retrieval (keyword)
    if bm25 is not None and rag_chunks:
        try:
            contexts_bm25, cites_bm25 = retrieve(question, rag_chunks, bm25, top_k=4)
        except Exception as e:
            print(f"[WARN] BM25 retrieve() failed: {e}")

    # 2) Vector retrieval (semantic)
    if vec_index is not None and rag_chunks:
        try:
            contexts_vec, cites_vec = retrieve_vec(question, rag_chunks, vec_index, top_k=4)
        except Exception as e:
            print(f"[WARN] Vector retrieve_vec() failed: {e}")

    # 3) Merge results: BM25 first then vectors; dedupe by chunk_id
    seen = set()
    for ctx, c in list(zip(contexts_bm25, cites_bm25)) + list(zip(contexts_vec, cites_vec)):
        cid = getattr(c, "chunk_id", None) or "unknown"
        if cid in seen:
            continue
        seen.add(cid)

        contexts.append(ctx)
        cites_out.append(
            {
                "source": getattr(c, "source", None),
                "page": getattr(c, "page", None),
                "chunk_id": getattr(c, "chunk_id", None),
                "snippet": getattr(c, "snippet", None),
            }
        )

    # 4) Clamp context (avoid slow prompts / stalls)
    contexts = contexts[:3]
    contexts = [c[:800] for c in contexts]


    # -----------------------------
    # Prompt build
    # -----------------------------
    if contexts:
        rag_block = "\n\n".join(contexts)
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"{mode_instructions(req.mode)}\n"
            f"Subject: {(req.subject or 'cs').strip().lower()}\n"
            f"Use the CONTEXT to answer. If the answer is not in the context, say you don't know.\n\n"
            f"CONTEXT:\n{rag_block}\n\n"
            f"User question: {question}\n"
            f"Answer (include short citations like [chunk_id] when you use context):\n"
            f"(Stop after finishing your answer.)\n"
        )
    else:
        prompt = build_prompt(req.subject, question, req.mode)

    # -----------------------------
    # LLM generate
    # -----------------------------
    print("ROUTING -> LLM (slow path)")
    out = llm(
        prompt,
        max_tokens=192,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>", "\n\nUser:", "\n\nUSER:", "\n\nQuestion:"],
        echo=False,
    )

    text = out["choices"][0]["text"].strip()

    return AskResponse(answer=text, citations=cites_out)
