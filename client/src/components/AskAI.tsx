import { useState, useRef, useEffect, useCallback, type MouseEvent } from "react";
import type { View } from "../App";
import {
  sendChatMessageStream,
  stopChatStream,
  uploadDocumentForAnalysis,
  clearChatSession,
  listChatSessions,
  getChatSessionHistory,
  type StreamEvent,
  type ChatSessionSummary,
} from "../api";
import { RichText } from "./RichText";
import { initials, avatarTint, type UserProfile } from "../lib/ui";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: StreamEvent;
  streaming?: boolean;
  error?: boolean;
  stopped?: boolean;
}

const EXAMPLE_PROMPTS = [
  "Draft a rent agreement",
  "Check bail eligibility",
  "File a consumer complaint",
  "Understand a legal notice",
  "Divorce process basics",
  "Trademark registration steps",
];

const CMDK_ACTIONS: { icon: string; label: string; view: View }[] = [
  { icon: "+", label: "New conversation", view: "chat" },
  { icon: "⚖", label: "Find Lawyers", view: "lawyers" },
  { icon: "📅", label: "My Bookings", view: "bookings" },
];

interface AskAIProps {
  user: UserProfile | null;
  initialQuestion: string | null;
  onConsumeInitial: () => void;
  onBookLawyer: () => void;
  onViewLawyer: () => void;
  onNavigate: (v: View) => void;
}

export function AskAI({ user, initialQuestion, onConsumeInitial, onNavigate }: AskAIProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [cmdkOpen, setCmdkOpen] = useState(false);
  const [cmdkQuery, setCmdkQuery] = useState("");
  const [openSources, setOpenSources] = useState<Record<string, boolean>>({});
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const endRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingSessionIdRef = useRef<string | undefined>(undefined);

  const greeting = (() => {
    const h = new Date().getHours();
    return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
  })();
  const firstName = user?.name?.split(" ")[0] ?? "there";

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const refreshSessions = useCallback(async () => {
    if (!user) {
      setSessions([]);
      return;
    }
    try {
      const { sessions } = await listChatSessions();
      setSessions(sessions);
    } catch {
      // Non-fatal — leave whatever list we already have.
    }
  }, [user]);

  // Covers initial mount, login, and logout. On logout, also drop any
  // transcript on screen so the next guest/account never sees it.
  useEffect(() => {
    refreshSessions();
    if (!user) {
      setSessionId(undefined);
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const loadSession = useCallback(
    async (id: string) => {
      if (id === sessionId || busy) return;
      setLoadingHistory(true);
      try {
        const { messages: history } = await getChatSessionHistory(id);
        setSessionId(id);
        setMessages(
          history.map((m, i) => ({
            id: `h${i}-${id}`,
            role: m.role === "assistant" ? "assistant" : "user",
            content: m.content,
          }))
        );
      } catch {
        // Leave the current view unchanged on failure.
      } finally {
        setLoadingHistory(false);
      }
    },
    [sessionId, busy]
  );

  const deleteSession = useCallback(
    async (e: MouseEvent, id: string) => {
      e.stopPropagation();
      try {
        await clearChatSession(id);
      } catch {
        // ignore
      }
      await refreshSessions();
      if (id === sessionId) {
        setSessionId(undefined);
        setMessages([]);
      }
    },
    [sessionId, refreshSessions]
  );

  const send = useCallback(
    async (text: string, upload?: File | null) => {
      if ((!text.trim() && !upload) || busy) return;
      const userMsg: Message = {
        id: `u${Date.now()}`,
        role: "user",
        content: upload ? `📎 ${upload.name}${text ? `\n${text}` : ""}` : text,
      };
      const botId = `a${Date.now()}`;
      setMessages((m) => [...m, userMsg, { id: botId, role: "assistant", content: "", streaming: true }]);
      setInput("");
      setFile(null);
      setBusy(true);

      try {
        if (upload) {
          const controller = new AbortController();
          abortRef.current = controller;
          try {
            const data = await uploadDocumentForAnalysis(
              upload,
              text || "Please analyze this document",
              sessionId,
              controller.signal
            );
            if (data.session_id) setSessionId(data.session_id);
            setMessages((m) => m.map((msg) => msg.id === botId && !msg.stopped ? {
              ...msg, streaming: false, content: data.response || "", meta: data as StreamEvent,
            } : msg));
          } catch (e: any) {
            if (e?.name !== "AbortError") throw e;
            // stopGenerating() already marked this message stopped; the
            // request was cancelled client-side, nothing more to update.
          }
        } else {
          // Known upfront (not just learned from the "done" event) so the
          // Stop button can cancel the server-side generation even if the
          // user clicks it before a single event has come back.
          const activeSessionId = sessionId ?? crypto.randomUUID();
          if (!sessionId) setSessionId(activeSessionId);
          streamingSessionIdRef.current = activeSessionId;

          const controller = new AbortController();
          abortRef.current = controller;

          let acc = "";
          await sendChatMessageStream(
            text,
            activeSessionId,
            (tok) => {
              acc += tok;
              setMessages((m) => m.map((msg) => msg.id === botId ? { ...msg, content: acc } : msg));
            },
            (meta) => {
              if (meta.session_id) setSessionId(meta.session_id);
              setMessages((m) => m.map((msg) => msg.id === botId ? {
                ...msg, streaming: false, meta, stopped: meta.type === "stopped",
              } : msg));
            },
            (err) => {
              setMessages((m) => m.map((msg) => msg.id === botId ? { ...msg, streaming: false, error: true, content: err } : msg));
            },
            controller.signal,
          );
        }
      } catch (e: any) {
        setMessages((m) => m.map((msg) => msg.id === botId ? { ...msg, streaming: false, error: true, content: e.message || "Something went wrong." } : msg));
      } finally {
        setBusy(false);
        abortRef.current = null;
        streamingSessionIdRef.current = undefined;
        if (user) refreshSessions();
      }
    },
    [busy, sessionId, user, refreshSessions]
  );

  const stopGenerating = useCallback(() => {
    abortRef.current?.abort();
    if (streamingSessionIdRef.current) stopChatStream(streamingSessionIdRef.current);
    setMessages((m) => m.map((msg) => msg.streaming ? { ...msg, streaming: false, stopped: true } : msg));
  }, []);

  // Consume a question passed in from the Home hero.
  useEffect(() => {
    if (initialQuestion) {
      send(initialQuestion);
      onConsumeInitial();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuestion]);

  // ⌘K
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCmdkOpen(true);
        setCmdkQuery("");
      } else if (e.key === "Escape") setCmdkOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const newConversation = async () => {
    if (sessionId) clearChatSession(sessionId).catch(() => {});
    setSessionId(undefined);
    setMessages([]);
  };

  const isEmpty = messages.length === 0;
  const q = cmdkQuery.toLowerCase();

  return (
    <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
      {/* sidebar */}
      <div style={{ width: 248, flex: "none", background: "var(--surface-alt)", borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", padding: "18px 16px" }} className="lw-sidebar">
        <button onClick={() => setCmdkOpen(true)} className="side-search">
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}><span>🔍</span>Search</span>
          <span className="kbd">⌘K</span>
        </button>
        <button onClick={newConversation} className="side-new">
          <span style={{ fontSize: 15, color: "var(--accent)" }}>+</span> New conversation
        </button>
        <div className="eyebrow" style={{ padding: "0 12px 8px", fontSize: 11 }}>Recent</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, overflow: "auto" }}>
          {user && sessions.length === 0 && (
            <div style={{ padding: "6px 12px", font: "400 12.5px var(--font-body)", color: "var(--muted-3)" }}>No conversations yet</div>
          )}
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => loadSession(s.id)}
              className={`side-thread${s.id === sessionId ? " active" : ""}`}
              style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}
            >
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.title || "Untitled"}</span>
              <span
                onClick={(e) => deleteSession(e, s.id)}
                className="side-thread-delete"
                style={{ flex: "none", opacity: 0, cursor: "pointer" }}
                title="Delete conversation"
              >
                ✕
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* main column */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {loadingHistory ? (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ display: "flex", gap: 4 }}>
              {[0, 1, 2].map((i) => (
                <span key={i} style={{ width: 6, height: 6, borderRadius: 99, background: "var(--faint)", display: "inline-block", animation: `lw-dotPulse 1.2s infinite ${i * 0.15}s` }} />
              ))}
            </div>
          </div>
        ) : isEmpty ? (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24, textAlign: "center", gap: 22 }}>
            <div style={{ width: 48, height: 48, borderRadius: 14, background: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", font: "700 20px var(--font-head)" }}>L</div>
            <div>
              <div style={{ font: "800 27px var(--font-head)" }}>{greeting}, {firstName}</div>
              <div style={{ font: "400 15px/1.6 var(--font-body)", color: "var(--muted-2)", marginTop: 8, maxWidth: 440 }}>
                Ask anything about Indian law in plain language — I'll cite the relevant sections and connect you with a lawyer if you need one.
              </div>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center", maxWidth: 560 }}>
              {EXAMPLE_PROMPTS.map((p) => (
                <button key={p} className="chip" onClick={() => send(p)}>{p}</button>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ flex: 1, overflow: "auto", display: "flex", justifyContent: "center" }}>
            <div style={{ width: "100%", maxWidth: 720, padding: "48px 24px 24px", display: "flex", flexDirection: "column", gap: 32 }}>
              {messages.map((m) => (m.role === "user" ? (
                <div key={m.id} style={{ alignSelf: "flex-end", maxWidth: "72%", background: "var(--user-bubble-bg)", color: "var(--user-bubble-fg)", padding: "14px 18px", borderRadius: 18, font: "400 15px/1.6 var(--font-body)", whiteSpace: "pre-wrap" }}>{m.content}</div>
              ) : (
                <div key={m.id} className="fade-up" style={{ display: "flex", gap: 12 }}>
                  <div style={{ width: 30, height: 30, flex: "none", borderRadius: 9, background: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", font: "700 13px var(--font-head)" }}>L</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ font: "600 13px var(--font-body)", color: "var(--muted-2)", marginBottom: 6 }}>LawWeb AI</div>
                    {m.streaming && !m.content ? (
                      <div style={{ display: "flex", gap: 4, padding: "10px 0" }}>
                        {[0, 1, 2].map((i) => (
                          <span key={i} style={{ width: 6, height: 6, borderRadius: 99, background: "var(--faint)", display: "inline-block", animation: `lw-dotPulse 1.2s infinite ${i * 0.15}s` }} />
                        ))}
                      </div>
                    ) : (
                      <div style={{ font: "400 16px/1.75 var(--font-body)", color: m.error ? "var(--danger)" : "var(--text-strong)" }}>
                        <RichText text={m.content} />
                      </div>
                    )}

                    {/* handoff: lawyers surfaced by the chatbot */}
                    {m.meta?.lawyers_found && m.meta.lawyers_found.length > 0 && (
                      <div style={{ marginTop: 16 }}>
                        <div style={{ font: "600 13px var(--font-body)", color: "var(--muted-2)", marginBottom: 10 }}>Lawyers who could help with this</div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                          {m.meta.lawyers_found.slice(0, 2).map((l: any, i: number) => {
                            const name = l.name || "Lawyer";
                            const tint = avatarTint(name);
                            return (
                              <div key={i} className="card" style={{ display: "flex", gap: 14, alignItems: "center" }}>
                                <div className="avatar" style={{ width: 46, height: 46, background: tint.bg, color: tint.fg }}>{initials(name)}</div>
                                <div style={{ flex: 1 }}>
                                  <div style={{ font: "700 14.5px var(--font-head)" }}>{name}</div>
                                  <div style={{ font: "500 12.5px var(--font-body)", color: "var(--muted-2)" }}>{l.specialization || l.specialty || "Legal"}{l.location ? ` · ${l.location}` : ""}</div>
                                </div>
                                <button className="btn btn-primary btn-sm" onClick={() => onNavigate("lawyers")}>Find lawyers</button>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* details from structured metadata (real, when present) */}
                    {(m.meta?.document_info || m.meta?.crime_report) && (
                      <div style={{ marginTop: 16 }}>
                        <button
                          onClick={() => setOpenSources((s) => ({ ...s, [m.id]: !s[m.id] }))}
                          style={{ display: "inline-flex", alignItems: "center", gap: 6, font: "500 13px var(--font-body)", color: "#7c8794", cursor: "pointer", background: "none", border: "none", padding: "4px 0" }}
                        >
                          <span style={{ display: "inline-block", transition: "transform .15s", transform: openSources[m.id] ? "rotate(90deg)" : "rotate(0deg)" }}>›</span>
                          {openSources[m.id] ? "Hide details" : "Show details"}
                        </button>
                        {openSources[m.id] && (
                          <pre style={{ marginTop: 8, background: "var(--surface-tint)", border: "1px solid var(--divider)", borderRadius: 12, padding: "14px 16px", font: "400 12.5px/1.6 var(--font-body)", color: "var(--muted)", whiteSpace: "pre-wrap", overflowX: "auto" }}>
                            {JSON.stringify(m.meta?.document_info || m.meta?.crime_report, null, 2)}
                          </pre>
                        )}
                      </div>
                    )}

                    {!m.streaming && !m.error && (
                      <div style={{ marginTop: 10, font: "400 12.5px var(--font-body)", color: "var(--muted-3)" }}>
                        {m.stopped ? "stopped · " : ""}
                        {m.meta?.intent ? `${m.meta.intent.replace(/_/g, " ")} · ` : ""}educational information, not legal advice
                      </div>
                    )}
                  </div>
                </div>
              )))}
              <div ref={endRef} />
            </div>
          </div>
        )}

        {/* composer */}
        <div style={{ display: "flex", justifyContent: "center", padding: "0 24px 22px", flex: "none" }}>
          <div style={{ width: "100%", maxWidth: 720 }}>
            {file && (
              <div style={{ display: "inline-flex", alignItems: "center", gap: 8, marginBottom: 8, padding: "6px 12px", background: "var(--accent-tint)", color: "var(--accent)", borderRadius: 99, font: "500 12.5px var(--font-body)" }}>
                📎 {file.name}
                <button onClick={() => setFile(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--accent)" }}>✕</button>
              </div>
            )}
            <form
              onSubmit={(e) => { e.preventDefault(); send(input, file); }}
              style={{ display: "flex", alignItems: "flex-end", gap: 10, background: "var(--surface)", border: "1px solid var(--border-2)", borderRadius: 18, padding: "12px 12px 12px 18px", boxShadow: "var(--shadow-soft)" }}
            >
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input, file); } }}
                placeholder="Describe your situation…"
                rows={1}
                style={{ flex: 1, border: "none", outline: "none", resize: "none", background: "transparent", font: "400 15px/1.5 var(--font-body)", color: "var(--text)", maxHeight: 160 }}
              />
              <input ref={fileRef} type="file" hidden accept=".pdf,.doc,.docx,.png,.jpg,.jpeg" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <button type="button" onClick={() => fileRef.current?.click()} title="Attach a document" style={{ width: 34, height: 34, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--muted-2)", cursor: "pointer", background: "none", border: "none", fontSize: 16 }}>📎</button>
              {busy ? (
                <button type="button" onClick={stopGenerating} className="composer-send" title="Stop generating" style={{ background: "var(--text-strong)" }}>
                  <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "#fff" }} />
                </button>
              ) : (
                <button type="submit" className="composer-send" title="Send">↑</button>
              )}
            </form>
            <div style={{ textAlign: "center", marginTop: 10, font: "400 12px var(--font-body)", color: "var(--faint)" }}>
              Educational information, not a substitute for a lawyer
            </div>
          </div>
        </div>
      </div>

      {/* ⌘K modal */}
      {cmdkOpen && (
        <div className="backdrop" style={{ paddingTop: "12vh" }} onClick={() => setCmdkOpen(false)}>
          <div className="modal" style={{ width: 560, maxWidth: "90vw" }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 18px", borderBottom: "1px solid var(--border)" }}>
              <span style={{ color: "var(--muted-3)" }}>🔍</span>
              <input autoFocus value={cmdkQuery} onChange={(e) => setCmdkQuery(e.target.value)} placeholder="Jump to bookings, lawyers, or a past thread…" style={{ flex: 1, border: "none", outline: "none", font: "400 15px var(--font-body)", color: "var(--text)" }} />
              <span className="kbd">Esc</span>
            </div>
            <div style={{ maxHeight: 360, overflow: "auto", padding: 8 }}>
              <div className="eyebrow" style={{ padding: "8px 12px 4px", fontSize: 11 }}>Quick actions</div>
              {CMDK_ACTIONS.filter((a) => a.label.toLowerCase().includes(q)).map((a) => (
                <button key={a.label} className="menu-row" onClick={() => { setCmdkOpen(false); if (a.label === "New conversation") newConversation(); else onNavigate(a.view); }}>
                  <span style={{ color: "var(--accent)", width: 16, display: "inline-block", textAlign: "center", marginRight: 8 }}>{a.icon}</span>{a.label}
                </button>
              ))}
              {sessions.length > 0 && <div className="eyebrow" style={{ padding: "12px 12px 4px", fontSize: 11 }}>Recent threads</div>}
              {sessions.filter((s) => (s.title || "").toLowerCase().includes(q)).map((s) => (
                <button key={s.id} className="menu-row" onClick={() => { setCmdkOpen(false); loadSession(s.id); }}>
                  <span style={{ color: "#c3c2bc", width: 16, display: "inline-block", textAlign: "center", marginRight: 8 }}>💬</span>{s.title || "Untitled"}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
