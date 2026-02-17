from __future__ import annotations

CHEM_SYSTEM = """You are an offline Chemistry tutor.
Goals:
- Explain clearly and step-by-step.
- Show units, formulas, and reasoning at every step.
- If user asks for dangerous synthesis or illicit instructions, refuse and redirect to safe learning.
- Always cite sources when provided (from retrieved context)."""

CS_SYSTEM = """You are an offline Computer Science tutor.
Goals:
- Explain clearly and step-by-step.
- Prefer small, concrete examples.
- If code is provided, explain what it does, find bugs, and propose fixes.
- Suggest edge cases or tests when relevant.
- Always cite sources when provided (from retrieved context)."""

ANSWER_STYLE = """Answer format:
1) Direct answer (1-2 sentences)
2) Step-by-step explanation
3) Quick check (1-2 questions or sanity checks)
4) Citations (if any)"""


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
            "- Ask 1-3 guiding questions to help the student discover the answer.\n"
            "- Do NOT give the final answer immediately.\n"
            "- If the student insists after 2 exchanges, give the full answer.\n"
        )
    if m == "hints":
        return (
            "Mode: HINTS\n"
            "- Give exactly ONE hint that nudges without spoiling.\n"
            "- Do not give the full solution unless explicitly asked.\n"
            "- End by asking: 'Want another hint or the full solution?'\n"
        )
    return (
        "Mode: QUIZ\n"
        "- Create exactly 3 practice questions on the topic.\n"
        "- Number them 1, 2, 3.\n"
        "- Do NOT include answers unless the student asks.\n"
    )


def build_prompt(
    subject: str,
    question: str,
    mode: str,
    context_blocks: list[dict] | None = None,
    history: list[dict] | None = None,
) -> str:
    """
    Build the full LLM prompt.

    Args:
        subject: "cs" or "chem"
        question: the current user question
        mode: explain | socratic | hints | quiz
        context_blocks: list of {"text": ..., "source": ..., "page": ...} dicts from RAG
        history: list of {"role": "user"|"assistant", "content": ...} dicts (most recent last)
    """
    system = CHEM_SYSTEM if subject.strip().lower() == "chem" else CS_SYSTEM

    # --- Context block ---
    context_text = "No external context provided."
    if context_blocks:
        lines = []
        for i, b in enumerate(context_blocks, start=1):
            src = b.get("source", "unknown")
            page = b.get("page", "?")
            text = b.get("text", b.get("snippet", ""))
            lines.append(f"[Source {i}] ({src} p{page})\n{text}")
        context_text = "\n\n".join(lines)

    # --- Conversation history (last 3 exchanges = 6 messages max) ---
    history_text = ""
    if history:
        recent = history[-6:]
        turns = []
        for h in recent:
            role = h.get("role", "user")
            content = h.get("content", "")
            label = "Student" if role == "user" else "Tutor"
            turns.append(f"{label}: {content}")
        history_text = "\n".join(turns)

    prompt = f"""<SYSTEM>
{system}
</SYSTEM>

<INSTRUCTIONS>
{mode_instructions(mode)}
{ANSWER_STYLE}
</INSTRUCTIONS>

<CONTEXT>
{context_text}
</CONTEXT>
"""

    if history_text:
        prompt += f"""
<CONVERSATION_HISTORY>
{history_text}
</CONVERSATION_HISTORY>
"""

    prompt += f"""
<USER_QUESTION>
{question.strip()}
</USER_QUESTION>

Tutor:"""

    return prompt
