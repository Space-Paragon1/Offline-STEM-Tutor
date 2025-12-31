                    Offline STEM Tutor
This project is an offline-first AI tutoring system designed to support STEM learning (currently Computer Science and Chemistry) without reliance on cloud-based APIs. The system integrates local large language model (LLM) inference, retrieval-augmented generation (RAG), and deterministic computational tools within an interactive, human-centered user interface.

The backend is implemented in Python (FastAPI) and runs a local LLM using llama.cpp. It combines free-form reasoning from the model with structured tools for exact scientific computation (e.g., molar mass and molarity calculations) and document-based retrieval using a BM25 index. This hybrid approach improves reliability, correctness, and transparency compared to purely generative systems.

The frontend is built with React and Tauri, providing a lightweight desktop interface optimized for low-resource environments. Users can interact with the tutor through multiple pedagogical modes (explain, socratic, hints, quiz), enabling adaptive and human-centered learning experiences. Retrieved context is surfaced with citations to support interpretability and trust.

Overall, this project explores the intersection of AI, machine learning, and human-computer interaction (HCI) in offline educational systems, with an emphasis on accessibility, explainability, and practical deployment on consumer-grade hardware.

Offline-first AI tutor for Computer Science and Chemistry, built with:
FastAPI (Python backend)
llama.cpp (local LLM, CPU-only)
BM25 Retrieval-Augmented Generation (RAG)
React + TypeScript frontend
Rule-based chemistry tools (molar mass, molarity)
No internet. No OpenAI API. Fully local.


✨ Features (Current)
✅ Core Tutor
Ask questions in Computer Science or Chemistry
Multiple tutor modes:
*explain – step-by-step explanation
*socratic – guided questions
*hints – single hint at a time
*quiz – practice questions


✅ Chemistry Tools (Fast, Deterministic)
Bypasses the LLM for accuracy and speed:
*Molar mass
  -Supports formulas like Ca(OH)₂, H₂SO₄, NaCl
*Molarity
  -From moles + volume
  -From mass + formula + volume

✅ Retrieval-Augmented Generation (RAG)
*BM25-based document search
*Chunked PDFs / TXT / MD files
*Context injection into the prompt
*Automatic citations returned to frontend

✅ Frontend UI
*Chat-style interface
*Subject + Mode selector
*Citation rendering under answers
*Clear / Send / Loading states

🏗️ Project Structure
offline-stem-tutor/
├── backend/
│   ├── main.py                 # FastAPI backend
│   ├── models/
│   │   └── model.gguf           # Local LLM (gguf)
│   ├── tools_chem.py            # Chemistry tools
│   ├── rag/
│   │   ├── ingest.py            # Document ingestion
│   │   └── retriever_bm25.py    # BM25 retrieval
│   └── data/
│       ├── docs/               # Source documents
│       └── index/
│           ├── chunks.jsonl
│           └── bm25.pkl
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   └── App.css
│   └── package.json
│
└── README.md

⚙️ Requirements
System
*Windows / macOS / Linux
*CPU-only (tested on Intel i5, 8GB RAM)
*~4–6 GB free RAM recommended

Backend
*Python 3.10+
*llama-cpp-python
*fastapi
*uvicorn

Frontend
*Node.js 18+
*Vite + React + TypeScript


🚀 Getting Started
1️⃣ Backend Setup
cd backend
pip install -r requirements.txt

place your GGUF model here:
backend/models/mode.gguf
Run the backend:
uvicorn main:app --host 127.0.0.1 --port 8123
check:
http://127.0.0.1:8123/health

2️⃣ Add Documents (RAG)
Put PDFs / TXT / MD files into:
backend/data/docs/
Then build the index:
POST http://127.0.0.1:8123/ingest

3️⃣ Frontend Setup
cd frontend
npm install
npm run dev
Open:
http://localhost:5173

🧪 Example Queries
Chemistry
*What is the molar mass of Ca(OH)2?
*What is the molarity of 0.5 mol in 250 mL?
*What is the molarity of 10 g NaCl in 500 mL?
Computer Science
*Explain recursion in Python
*Give me hints for binary search
*Quiz me on linked lists

🧠 Design Philosophy
Offline-first → works without internet
Tools > LLM when accuracy matters
RAG for grounded answers
Small context windows for stability on low-RAM systems
Explicit stop conditions to prevent infinite generation

🗺️ Roadmap
✅ Completed
Phase 1 – Backend + frontend
Phase 2 – Tutor modes
Phase 3 – Chemistry tools
Phase 4A – BM25 RAG + citations
🔜 Next
Phase 4B – Embeddings (semantic search)
Phase 5 – Saved chats & offline memory
Phase 6 – Exam mode (timed quizzes)