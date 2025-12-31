CHEM_SYSTEM = """You are an offline Chemistry tutor.
Goals:
- Explain clearly and step-by-step.
- Ask 1 clarifying question only if absolutely necessary.
- Show units, formulas, and reasoning.
- If user asks for dangerous synthesis or illicit instructions, refuse and redirect to safe learning.
- Always cite sources when provided (from retrieved notes)."""

CS_SYSTEM = """You are an offline Computer Science tutor.
Goals:
- Explain clearly and step-by-step.
- Prefer small examples.
- If code is provided, explain what it does, find bugs, and propose fixes.
- Suggest tests.
- Always cite sources when provided (from retrieved notes)."""

ANSWER_STYLE = """Answer format:
1) Direct answer
2) Step-by-step explanation
3) Quick check (1-2 questions or sanity checks)
4) Citations (if any)
"""

def build_prompt(subject: str, question: str, context_blocks: list[dict]) -> str:
    system = CHEM_SYSTEM if subject.lower() == "chem" else CS_SYSTEM
    context_text = ""
    if context_blocks:
        lines = []
        for i, b in enumerate(context_blocks, start=1):
            lines.append(f"[Source {i}] ({b.get('source','unknown')} p{b.get('page','?')})\n{b['text']}\n")
        context_text = "\n\n".join(lines)

    prompt = f"""<SYSTEM>
{system}
</SYSTEM>

<CONTEXT>
{context_text if context_text else "No external context provided."}
</CONTEXT>

<INSTRUCTIONS>
{ANSWER_STYLE}
</INSTRUCTIONS>

<USER_QUESTION>
{question}
</USER_QUESTION>
"""
    return prompt
