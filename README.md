# Offline STEM Tutor

An offline-first AI tutoring system for **Computer Science** and **Chemistry**, built to run entirely on consumer hardware with no cloud APIs or internet required.

The backend uses a local LLM (Phi-3 Mini 3.8B via llama.cpp) combined with hybrid RAG retrieval (BM25 + vector search) and deterministic chemistry tools. The frontend is a lightweight React + Tauri desktop app with markdown-rendered responses and conversation history.

## Features

### Core Tutor
- Ask questions in **Computer Science** or **Chemistry**
- Four pedagogical modes:
  - **Explain** -- step-by-step explanation with examples
  - **Socratic** -- guided questions to help discover the answer
  - **Hints** -- one hint at a time without spoiling
  - **Quiz** -- 3 practice questions on the topic
- **Conversation history** -- the tutor remembers the last 3 exchanges for multi-turn dialogue
- **Markdown rendering** -- responses display formatted text, code blocks, tables, and lists

### Chemistry Tools (Deterministic)
Bypasses the LLM for accuracy and speed:
- **Molar mass** -- supports formulas like `Ca(OH)2`, `H2SO4`, `NaCl`
- **Molarity** -- from moles + volume, or from mass + formula + volume
- Improved formula extraction with scoring (prefers tokens containing digits/parentheses)

### Retrieval-Augmented Generation (RAG)
- **Hybrid search**: BM25 (keyword) + sentence-transformers (semantic/vector)
- Chunked PDFs, TXT, and MD documents
- Context injection into the LLM prompt
- Citations returned to the frontend with source and page info

### Frontend UI
- Chat-style interface with subject and mode selectors
- Markdown-rendered assistant responses (via react-markdown + remark-gfm)
- Citation rendering under answers
- "Thinking..." animation during LLM inference
- Clear / Send / Loading states

## Project Structure

```
AI-Stem-tutor/
├── backend/
│   ├── main.py                  # FastAPI app, routes, LLM loader
│   ├── requirements.txt         # Python dependencies (pinned)
│   ├── tools_chem.py            # Molar mass & molarity tools
│   ├── tools_cs.py              # CS tools (placeholder)
│   ├── models/
│   │   └── model.gguf           # Local GGUF model (Phi-3 Mini Q4_K_M)
│   ├── tutor/
│   │   ├── prompts.py           # Prompt builder (system prompts, modes, history)
│   │   ├── llm.py               # LLM utilities
│   │   ├── ingest.py            # Tutor-level ingestion helpers
│   │   └── retriever.py         # Tutor-level retrieval helpers
│   ├── rag/
│   │   ├── ingest.py            # Document chunking (PDF/TXT/MD)
│   │   ├── retriever_bm25.py    # BM25 keyword retrieval
│   │   └── retriever_vec.py     # Vector/semantic retrieval
│   └── data/
│       ├── docs/                # Source documents (place PDFs here)
│       └── index/
│           ├── chunks.jsonl     # Chunked document index
│           ├── bm25.pkl         # BM25 index
│           └── vec.pkl          # Vector index (after /ingest)
│
├── src/
│   ├── App.tsx                  # Main React component
│   ├── api.ts                   # API client (askTutor, types)
│   ├── App.css                  # Styles (markdown prose, animations)
│   └── main.tsx                 # React entry point
│
├── src-tauri/                   # Tauri 2 desktop wrapper config
├── package.json                 # Frontend dependencies
├── vite.config.ts               # Vite config
└── README.md
```

## Requirements

### System
- Windows / macOS / Linux
- CPU-only (tested on Intel i5, 8 GB RAM)
- **4-6 GB free RAM** required to load the model

### Backend
- Python 3.10+
- Key packages: `fastapi`, `uvicorn`, `llama-cpp-python`, `rank-bm25`, `sentence-transformers`

### Frontend
- Node.js 18+
- Vite + React 19 + TypeScript

## Getting Started

### 1. Backend Setup

```bash
cd backend

# Install dependencies (use the project venv)
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt # macOS/Linux
```

Place a GGUF model file at `backend/models/model.gguf`. The project uses **Phi-3 Mini 3.8B Q4_K_M** (~2.2 GB).

Start the backend:

```bash
.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8123 --reload
```

Verify it's running:

```
GET http://127.0.0.1:8123/health
```

The health endpoint reports model status, loaded chunks, and index state.

### 2. Add Documents (RAG)

Place PDF, TXT, or MD files into `backend/data/docs/`, then build the index:

```
POST http://127.0.0.1:8123/ingest
```

This chunks the documents and builds both BM25 and vector indexes.

### 3. Frontend Setup

From the project root:

```bash
npm install
npm run dev
```

Open: **http://localhost:5173**

## Example Queries

### Chemistry
- "What is the molar mass of Ca(OH)2?"
- "What is the molarity of 0.5 mol in 250 mL?"
- "What is the molarity of 10 g NaCl in 500 mL?"

### Computer Science
- "Explain recursion in Python"
- "Give me hints for binary search"
- "Quiz me on linked lists"

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | System status, model info, index state |
| POST   | `/ingest` | Chunk documents and build RAG indexes |
| POST   | `/ask`    | Ask a question (returns answer + citations) |

### POST /ask

```json
{
  "question": "What are tuples?",
  "subject": "cs",
  "mode": "explain",
  "history": [
    { "role": "user", "content": "What is a list?" },
    { "role": "assistant", "content": "A list is a mutable ordered collection..." }
  ]
}
```

Response:

```json
{
  "answer": "A tuple is an immutable ordered collection...",
  "citations": [
    { "source": "python-basics.pdf", "page": 12, "chunk_id": "c3", "snippet": "..." }
  ]
}
```

## Design Philosophy

- **Offline-first** -- works without internet or cloud APIs
- **Tools over LLM** when deterministic accuracy matters (chemistry calculations)
- **Hybrid RAG** for grounded, cited answers
- **Conversation history** for coherent multi-turn tutoring
- **Small context windows** for stability on low-RAM systems
- **Explicit stop conditions** to prevent runaway generation

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Failed to create llama_context` | Close apps to free RAM (need 4+ GB free) |
| `LLM not loaded` | Check `/health` -- model file may be missing or RAM insufficient |
| `Failed to fetch` in frontend | Backend not running or still reloading after file change |
| Truncated responses | Increase `max_tokens` in `main.py` (currently 350) |

## Roadmap

### Completed
- Phase 1 -- Backend + frontend
- Phase 2 -- Tutor modes (explain, socratic, hints, quiz)
- Phase 3 -- Chemistry tools (molar mass, molarity)
- Phase 4A -- BM25 RAG + citations
- Phase 4B -- Vector/semantic search (sentence-transformers)
- Conversation history support
- Markdown rendering in frontend
- Lifespan context manager (replaced deprecated startup events)

### Next
- Phase 5 -- Saved chats and offline memory
- Phase 6 -- Exam mode (timed quizzes)
- Physics subject support
- Export chat as PDF
