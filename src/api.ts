// src/api.ts
export type Subject = "cs" | "chem";
export type Mode = "explain" | "socratic" | "hints" | "quiz";

export type HistoryItem = {
  role: "user" | "assistant";
  content: string;
};

export type Citation = {
  source?: string;
  page?: number;
  chunk_id?: string;
  snippet?: string;
};

export type AskResponse = {
  answer: string;
  citations: Citation[];
};

export async function askTutor(
  question: string,
  subject: Subject,
  mode: Mode,
  history: HistoryItem[] = []
): Promise<AskResponse> {
  const res = await fetch("http://127.0.0.1:8123/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, subject, mode, history }),
  });

  let data: any = null;
  try {
    data = await res.json();
  } catch {
    // ignore parse errors
  }

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.message)) ||
      `Backend error (${res.status})`;
    throw new Error(msg);
  }

  return {
    answer: data?.answer ?? "",
    citations: data?.citations ?? [],
  };
}
