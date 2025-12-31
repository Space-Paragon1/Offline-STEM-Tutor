// src/api.ts
export type Subject = "cs" | "chem";
export type Mode = "explain" | "socratic" | "hints" | "quiz";

export type AskResponse = {
  answer: string;
  citations?: Array<{
    source?: string;
    text?: string;
    start?: number;
    end?: number;
  }>;
};

export async function askTutor(
  question: string,
  subject: Subject,
  mode: Mode
): Promise<AskResponse> {
  const res = await fetch("http://127.0.0.1:8123/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, subject, mode }),
  });

  // Try to parse JSON even on error (FastAPI often sends JSON errors)
  let data: any = null;
  try {
    data = await res.json();
  } catch {
    // ignore
  }

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.message)) ||
      `Backend error (${res.status})`;
    throw new Error(msg);
  }

  // Ensure shape
  return {
    answer: data?.answer ?? "",
    citations: data?.citations ?? [],
  };
}
