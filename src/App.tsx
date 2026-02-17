import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";
import { askTutor, Subject, Mode, HistoryItem, Citation } from "./api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  subject?: Subject;
  citations?: Citation[];
};

export default function App() {
  const [subject, setSubject] = useState<Subject>("cs");
  const [mode, setMode] = useState<Mode>("explain");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm your offline STEM tutor. Pick Chemistry or CS and ask a question.",
    },
  ]);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const canSend = useMemo(
    () => question.trim().length > 0 && !isLoading,
    [question, isLoading]
  );

  async function onSend() {
    if (!canSend) return;

    const q = question.trim();
    setQuestion("");
    setError("");

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: q, subject },
    ];
    setMessages(nextMessages);

    // Build history from all prior turns (exclude initial greeting, exclude last user msg)
    const history: HistoryItem[] = nextMessages
      .slice(0, -1) // exclude the message we just added
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));

    setIsLoading(true);
    try {
      const res = await askTutor(q, subject, mode, history);

      const answerText =
        res.answer?.trim().length > 0
          ? res.answer
          : "(No answer returned from backend — check backend response format)";

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answerText,
          citations: res.citations ?? [],
        },
      ]);
    } catch (e: any) {
      setError(
        e?.message ||
          "Could not reach the Python backend. Make sure it's running on http://127.0.0.1:8123"
      );
    } finally {
      setIsLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void onSend();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="title">Offline STEM Tutor</div>
          <div className="subtitle">
            Tauri + React UI → Python (FastAPI) backend
          </div>
        </div>

        <div className="controls">
          <label className="label">
            Subject
            <select
              className="select"
              value={subject}
              onChange={(e) => setSubject(e.target.value as Subject)}
              disabled={isLoading}
            >
              <option value="cs">Computer Science</option>
              <option value="chem">Chemistry</option>
            </select>
          </label>

          <label className="label">
            Mode
            <select
              className="select"
              value={mode}
              onChange={(e) => setMode(e.target.value as Mode)}
              disabled={isLoading}
            >
              <option value="explain">Explain</option>
              <option value="socratic">Socratic</option>
              <option value="hints">Hints</option>
              <option value="quiz">Quiz</option>
            </select>
          </label>
        </div>
      </header>

      <main className="chat">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`msg ${m.role === "user" ? "msgUser" : "msgAssistant"}`}
          >
            <div className="msgRole">
              {m.role === "user" ? "You" : "Tutor"}
              {m.role === "user" && m.subject ? (
                <span className="pill">{m.subject.toUpperCase()}</span>
              ) : null}
            </div>

            <div className="msgContent">
              {m.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {m.content}
                </ReactMarkdown>
              ) : (
                m.content
              )}
            </div>

            {m.role === "assistant" &&
              m.citations &&
              m.citations.length > 0 && (
                <div className="citations">
                  <div className="citationsTitle">Sources</div>
                  <ul className="citationsList">
                    {m.citations.map((c, i) => (
                      <li key={i} className="citationItem">
                        <div className="citationMeta">
                          <span className="citationSource">
                            {c.source ?? "unknown"}
                          </span>
                          {typeof c.page === "number" && c.page > 0 ? (
                            <span className="citationPage"> p.{c.page}</span>
                          ) : null}
                          {c.chunk_id ? (
                            <span className="citationChunk">
                              {" "}[{c.chunk_id}]
                            </span>
                          ) : null}
                        </div>
                        {c.snippet ? (
                          <div className="citationSnippet">{c.snippet}</div>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
          </div>
        ))}

        {isLoading && (
          <div className="msg msgAssistant">
            <div className="msgRole">Tutor</div>
            <div className="msgContent thinking">Thinking…</div>
          </div>
        )}
      </main>

      <footer className="composer">
        {error ? <div className="error">{error}</div> : null}

        <textarea
          className="textarea"
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={isLoading}
        />

        <div className="actions">
          <button
            className="button"
            onClick={() => setMessages(messages.slice(0, 1))}
            disabled={isLoading}
          >
            Clear
          </button>

          <button
            className="button primary"
            onClick={onSend}
            disabled={!canSend}
          >
            Send
          </button>
        </div>

        <div className="hint">
          Backend must be running at <code>http://127.0.0.1:8123</code>
        </div>
      </footer>
    </div>
  );
}
