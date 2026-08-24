import { useState, useEffect } from "react";
import type { View } from "../App";
import { fetchUserBookings, fetchLawyers, type Booking } from "../api";
import { initials, avatarTint, formatMoney, type Lawyer, type UserProfile } from "../lib/ui";

interface Props {
  user: UserProfile | null;
  onNavigate: (v: View) => void;
}

function formatDate(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-IN", { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export function MyBookings({ user, onNavigate }: Props) {
  const [bookings, setBookings] = useState<Booking[]>([]);
  const [lawyers, setLawyers] = useState<Record<string, Lawyer>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) {
      // Reachable if a session expires while this view is already open —
      // without this the skeleton loader would spin forever instead of
      // ever resolving.
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.all([fetchUserBookings(user.id), fetchLawyers().catch(() => [])])
      .then(([bk, lw]) => {
        setBookings(bk);
        const map: Record<string, Lawyer> = {};
        (lw as Lawyer[]).forEach((l) => (map[l.id] = l));
        setLawyers(map);
      })
      .catch(() => setError("Couldn't load your bookings."))
      .finally(() => setLoading(false));
  }, [user]);

  const totalSpent = bookings.reduce((sum, b) => sum + (b.amount || 0), 0);
  const stats = [
    { label: "Total consultations", value: String(bookings.length) },
    { label: "Confirmed", value: String(bookings.filter((b) => b.status === "confirmed").length), accent: true },
    { label: "Total spent", value: formatMoney(totalSpent) },
  ];

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 820 }}>
        <h1 className="page-title">My bookings</h1>
        <p className="page-sub">Consultations you've scheduled with lawyers on LawWeb.</p>

        {/* stats strip */}
        <div className="grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 32 }}>
          {stats.map((s) => (
            <div key={s.label} className="card" style={{ padding: "16px 18px", borderRadius: 14 }}>
              <div style={{ font: "600 12px var(--font-body)", color: "var(--muted-3)" }}>{s.label}</div>
              <div style={{ font: "700 24px var(--font-head)", marginTop: 4, color: s.accent ? "var(--accent)" : "var(--text)" }}>{s.value}</div>
            </div>
          ))}
        </div>

        {error && <div className="card" style={{ borderColor: "#f0d9d6", color: "var(--danger)", marginBottom: 18 }}>{error}</div>}

        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {Array.from({ length: 2 }).map((_, i) => <div key={i} className="shimmer" style={{ height: 96, borderRadius: 16 }} />)}
          </div>
        ) : bookings.length === 0 ? (
          <div className="card" style={{ padding: 40, textAlign: "center", borderStyle: "dashed" }}>
            <div style={{ font: "600 15px var(--font-body)", color: "var(--text-2)", marginBottom: 6 }}>No consultations yet</div>
            <div style={{ font: "400 13.5px var(--font-body)", color: "var(--muted-2)", marginBottom: 18 }}>Ask a question first — we'll suggest lawyers who fit your case.</div>
            <button className="btn btn-primary" onClick={() => onNavigate("chat")}>Ask the Legal AI</button>
          </div>
        ) : (
          <>
            <div className="section-label">Consultations</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {bookings.map((b) => {
                const lawyer = lawyers[b.lawyerId];
                const name = lawyer?.name ?? "Legal consultation";
                const tint = avatarTint(name);
                return (
                  <div key={b.id} className="card" style={{ padding: "18px 20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
                      <div className="avatar" style={{ width: 44, height: 44, background: tint.bg, color: tint.fg }}>{initials(name)}</div>
                      <div style={{ flex: 1, minWidth: 160 }}>
                        <div style={{ font: "700 14.5px var(--font-head)" }}>{name}</div>
                        <div style={{ font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>{formatDate(b.createdAt)}{lawyer ? ` · ${lawyer.specialty}` : ""}</div>
                      </div>
                      <div style={{ font: "600 14px var(--font-body)" }}>{formatMoney(b.amount)}</div>
                      <span className="pill pill-green" style={{ textTransform: "capitalize" }}>{b.status}</span>
                    </div>
                    <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--divider)", font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>
                      Transaction: <span style={{ color: "var(--text-2)", fontWeight: 500 }}>{b.transactionId}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
