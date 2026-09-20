import { useEffect, useState } from "react";
import {
  connectCalendarProvider,
  createCalendarEvent,
  deleteCalendarEvent,
  disconnectCalendarProvider,
  fetchCalendarEvents,
  fetchCalendarProviders,
  syncCalendarProvider,
  type CalendarEvent,
  type CalendarProviderStatus,
} from "../api";
import type { UserProfile } from "../lib/ui";

interface Props {
  user: UserProfile | null;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export function LegalCalendar({ user }: Props) {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [title, setTitle] = useState("");
  const [startAt, setStartAt] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showPast, setShowPast] = useState(false);
  const [providers, setProviders] = useState<CalendarProviderStatus[]>([]);

  const loadProviders = () => {
    fetchCalendarProviders().then(setProviders).catch(() => {});
  };

  // The OAuth round-trip returns here as #/calendar?calendar_connected=google
  // (or calendar_error=google).
  useEffect(() => {
    const query = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
    const connected = query.get("calendar_connected");
    const failed = query.get("calendar_error");
    if (connected) setNotice(`Connected ${connected === "google" ? "Google Calendar" : "Outlook"}. Use "Sync now" to push your events.`);
    if (failed) setError(`Couldn't connect ${failed === "google" ? "Google Calendar" : "Outlook"}. Please try again.`);
    if (connected || failed) window.history.replaceState(null, "", "#/calendar");
  }, []);

  const load = () => {
    fetchCalendarEvents().then(setEvents).catch(() => setError("Couldn't load your calendar.")).finally(() => setLoading(false));
  };

  useEffect(() => { if (user) { load(); loadProviders(); } }, [user]);

  const handleAdd = async () => {
    if (!title.trim() || !startAt) return;
    try {
      await createCalendarEvent({ title, eventType: "custom", startAt: new Date(startAt).toISOString() });
      setTitle("");
      setStartAt("");
      load();
    } catch {
      setError("Couldn't add that event.");
    }
  };

  const attempt = async (action: () => Promise<unknown>, fallback: string, success?: string) => {
    setError(null);
    setNotice(null);
    try {
      await action();
      if (success) setNotice(success);
    } catch (e) {
      setError(e instanceof Error && e.message ? e.message : fallback);
    }
  };

  const connect = (provider: string) =>
    attempt(async () => { window.location.href = await connectCalendarProvider(provider); }, "Couldn't start the connection.");

  const syncNow = (provider: string) =>
    attempt(
      async () => {
        const r = await syncCalendarProvider(provider);
        setNotice(`Pushed ${r.pushed} event${r.pushed === 1 ? "" : "s"}${r.failed ? ` (${r.failed} failed)` : ""}.`);
        loadProviders();
      },
      "Sync failed.",
    );

  const disconnect = (provider: string) =>
    attempt(async () => { await disconnectCalendarProvider(provider); loadProviders(); }, "Couldn't disconnect.", "Disconnected.");

  if (!user) {
    return <div className="container"><p className="page-sub">Sign in to view your legal calendar.</p></div>;
  }

  const now = Date.now();
  const visibleEvents = events.filter((e) => (new Date(e.startAt).getTime() >= now) !== showPast);

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 780 }}>
        <h1 className="page-title">Personal Legal Calendar</h1>
        <p className="page-sub">Hearings, deadlines, and lawyer meetings — hearings and confirmed bookings are added automatically.</p>

        <div className="card" style={{ padding: 20, marginBottom: 24, display: "flex", gap: 10, flexWrap: "wrap" }}>
          <input className="input" style={{ flex: 1, minWidth: 200 }} placeholder="Add a deadline or reminder…" value={title} onChange={(e) => setTitle(e.target.value)} />
          <input className="input" type="datetime-local" value={startAt} onChange={(e) => setStartAt(e.target.value)} />
          <button className="btn btn-primary" onClick={handleAdd}>Add</button>
        </div>

        {providers.some((p) => p.configured) && (
          <div className="card" style={{ padding: 16, marginBottom: 20, display: "flex", flexDirection: "column", gap: 10 }}>
            <div className="section-label" style={{ margin: 0 }}>External calendars</div>
            {providers.filter((p) => p.configured).map((p) => (
              <div key={p.provider} style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <span style={{ font: "600 13.5px var(--font-body)", minWidth: 120 }}>{p.provider === "google" ? "Google Calendar" : "Outlook"}</span>
                {p.connected ? (
                  <>
                    <span style={{ font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>
                      {p.lastSyncedAt ? `Last synced ${new Date(p.lastSyncedAt).toLocaleString("en-IN")}` : "Connected"}
                    </span>
                    <button className="btn btn-sm" onClick={() => syncNow(p.provider)}>Sync now</button>
                    <button className="btn btn-ghost btn-sm" onClick={() => disconnect(p.provider)}>Disconnect</button>
                  </>
                ) : (
                  <button className="btn btn-outline btn-sm" onClick={() => connect(p.provider)}>Connect</button>
                )}
              </div>
            ))}
          </div>
        )}

        {notice && <div className="card" style={{ padding: 12, marginBottom: 18 }}>{notice}</div>}
        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}

        <div className="segmented" style={{ marginBottom: 16 }}>
          {([false, true] as const).map((past) => (
            <button key={String(past)} className={showPast === past ? "is-on" : ""} onClick={() => setShowPast(past)}>
              {past ? "Past" : "Upcoming"}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="shimmer" style={{ height: 140, borderRadius: "var(--r-lg)" }} />
        ) : visibleEvents.length === 0 ? (
          <div className="empty-state"><div className="empty-sub">{showPast ? "No past events." : "No upcoming events."}</div></div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {visibleEvents.map((e) => (
              <div key={e.id} className="card" style={{ padding: "14px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ font: "600 12px var(--font-body)", textTransform: "capitalize", color: "var(--accent)" }}>{e.eventType.replaceAll("_", " ")}</div>
                  <div style={{ font: "600 13.5px var(--font-body)" }}>{e.title}</div>
                  <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{formatDate(e.startAt)}</div>
                </div>
                <button className="btn btn-sm" onClick={() => attempt(() => deleteCalendarEvent(e.id), "Couldn't remove that event.").then(load)}>Remove</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
