import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, runToCompletion } from "../api/client";
import type { ChatMessage, ConversationRef, Evidence, ProgressEvent } from "../api/types";
import { ProgressLine } from "../components/Controls";
import { CitedText, EvidenceCard, SourcePanel } from "../components/Evidence";
import { LANGUAGE_NAMES, useI18n } from "../i18n";
import { useSession } from "../session";

export function ChatView() {
  const { t } = useI18n();
  const { me, answerLanguage } = useSession();
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<ConversationRef[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [reasoned, setReasoned] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [focus, setFocus] = useState<string | null>(null);
  const [open, setOpen] = useState<Evidence | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const refreshList = () =>
    api.get<{ conversations: ConversationRef[] }>("/api/conversations").then((r) => setConversations(r.conversations));

  useEffect(() => {
    refreshList().catch(() => {});
  }, []);

  useEffect(() => {
    setOpen(null);
    if (!sessionId) {
      setMessages([]);
      return;
    }
    if (busy) return; // a turn in progress already shows this conversation
    api
      .get<{ messages: ChatMessage[] }>(`/api/conversations/${sessionId}`)
      .then((r) => setMessages(r.messages))
      .catch(() => setMessages([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Braces matter: Chromium's scrollIntoView now returns a Promise, which React would take for a cleanup function
  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [messages, busy]);

  const send = async () => {
    const question = prompt.trim();
    if (!question || busy) return;
    setPrompt("");
    setBusy(true);
    setProgress([]);
    setMessages((m) => [...m, { role: "user", text: question }]);
    try {
      const result = await runToCompletion<{ session_id: string; message: ChatMessage }>(
        "/api/chat",
        { prompt: question, language: answerLanguage, mode: reasoned ? "reasoned" : "chat", session_id: sessionId },
        (e) => setProgress((p) => [...p, e]),
      );
      setMessages((m) => [...m, result.message]);
      if (result.session_id !== sessionId) navigate(`/chat/${result.session_id}`, { replace: !sessionId });
      refreshList().catch(() => {});
    } catch (e: any) {
      setMessages((m) => [...m, { role: "assistant", text: "", error: e.message ?? t("Something went wrong.") }]);
    } finally {
      setBusy(false);
    }
  };

  // [Doc N] numbers are conversation-wide: the newest card with that number is the one an answer means
  const cite = (n: number) => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const card = messages[i].evidence?.find((e) => e.doc_number === n);
      if (card) {
        setFocus(card.chunk_id);
        if (card.has_source) setOpen(card);
        document.getElementById(`doc-${card.chunk_id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
    }
  };

  const remove = async (id: string) => {
    await api.del(`/api/conversations/${id}`);
    if (id === sessionId) navigate("/chat");
    refreshList().catch(() => {});
  };

  return (
    <div className="panes">
      <aside className="pane pane-list">
        <h3>{t("Conversations")}</h3>
        <button className="primary wide" onClick={() => navigate("/chat")}>{t("New conversation")}</button>
        <ul className="history">
          {conversations.length === 0 && <li className="empty-note">{t("No conversations yet.")}</li>}
          {conversations.map((c) => (
            <li key={c.session_id} className={c.session_id === sessionId ? "active" : undefined}>
              <button onClick={() => navigate(`/chat/${c.session_id}`)} title={c.title}>
                <span className="title">{c.title}</span>
              </button>
              <button className="icon" aria-label={t("Delete")} onClick={() => remove(c.session_id)}>×</button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="pane pane-main">
        <div className="page-header">
          <h2>{t("Chat")}</h2>
          <span className="spacer" />
          <span className="subtle">{t("Answers in")} {LANGUAGE_NAMES[answerLanguage]}</span>
        </div>
        <div className="transcript">
          {messages.length === 0 && !busy && (
            <div className="empty-state">
              <h2>{me.firm?.name}</h2>
              <p>{t("Every claim is cited. Click a [Doc N] to see the excerpt it rests on.")}</p>
            </div>
          )}
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="turn-user">{m.text}</div>
            ) : (
              <div key={i} className="turn-assistant">
                <div className="speaker">Izvoria</div>
                {m.error ? (
                  <p className="banner danger">{m.error}</p>
                ) : (
                  <>
                    {m.outcome === "needs_clarification" && (
                      <p className="banner warning">{t("The system needs more detail:")}</p>
                    )}
                    {m.critique_unresolved && (
                      <p className="banner warning">{t("This answer could not be fully verified against the evidence.")}</p>
                    )}
                    <CitedText text={m.text} onCite={cite} />
                    {m.evidence && m.evidence.length > 0 && (
                      <details open className="evidence-list">
                        <summary>{t("Evidence")} ({m.evidence.length})</summary>
                        {m.evidence.map((e) => (
                          <EvidenceCard key={e.chunk_id + e.doc_number} evidence={e} highlighted={focus === e.chunk_id}
                                        selected={open?.chunk_id === e.chunk_id} onOpen={setOpen} />
                        ))}
                      </details>
                    )}
                  </>
                )}
              </div>
            ),
          )}
          {busy && <ProgressLine events={progress} fallback="Thinking…" />}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={t("Ask a question about your material…")}
            maxLength={4000}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <div className="composer-controls">
            <select value={reasoned ? "reasoned" : "chat"} onChange={(e) => setReasoned(e.target.value === "reasoned")}
                    title={t("Several search passes and a self-check. Slower, more thorough.")} aria-label={t("Answer mode")}>
              <option value="chat">{t("Quick answer")}</option>
              <option value="reasoned">{t("Thorough answer")}</option>
            </select>
            <span className="subtle">{t("Enter to send, Shift+Enter for a new line")}</span>
            <button className="primary" type="submit" disabled={busy || !prompt.trim()}>{t("Ask")}</button>
          </div>
        </form>
      </section>

      <SourcePanel evidence={open} onClose={() => setOpen(null)} />
    </div>
  );
}
