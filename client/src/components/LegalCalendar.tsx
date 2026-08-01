import { useEffect, useState } from "react";
import { createCalendarEvent, deleteCalendarEvent, fetchCalendarEvents, type CalendarEvent } from "../api";
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

  const load = () => {
    fetchCalendarEvents().then(setEvents).catch(() => setError("Couldn't load your calendar.")).finally(() => setLoading(false));
  };

  useEffect(() => { if (user) load(); }, [user]);

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

  if (!user) {
    return <div className="container"><p className="page-sub">Sign in to view your legal calendar.</p></div>;
  }

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

        {error && <div className="card" style={{ borderColor: "#f0d9d6", color: "var(--danger)", marginBottom: 18 }}>{error}</div>}

        {loading ? (
          <div className="shimmer" style={{ height: 140, borderRadius: 16 }} />
        ) : events.length === 0 ? (
          <div className="card" style={{ padding: 40, textAlign: "center", borderStyle: "dashed" }}>No upcoming events.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {events.map((e) => (
              <div key={e.id} className="card" style={{ padding: "14px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ font: "600 12px var(--font-body)", textTransform: "capitalize", color: "var(--accent)" }}>{e.eventType.replace("_", " ")}</div>
                  <div style={{ font: "600 13.5px var(--font-body)" }}>{e.title}</div>
                  <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{formatDate(e.startAt)}</div>
                </div>
                <button className="btn btn-sm" onClick={() => deleteCalendarEvent(e.id).then(load)}>Remove</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
