import { initials, avatarTint, availabilityColor, formatRate, type Lawyer } from "../lib/ui";

interface Props {
  lawyer: Lawyer;
  onBook: (l: Lawyer) => void;
  onBack: () => void;
}

export function LawyerProfile({ lawyer, onBook, onBack }: Props) {
  const tint = avatarTint(lawyer.name);
  const availColor = availabilityColor(lawyer.availability);
  const stats = [
    { label: "Experience", value: `${lawyer.experience} yrs` },
    { label: "Cases handled", value: lawyer.cases?.toLocaleString("en-IN") ?? "—" },
    { label: "Success rate", value: `${lawyer.successRate}%` },
    { label: "Rating", value: `★ ${lawyer.rating}` },
  ];

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 820 }}>
        <button className="btn btn-ghost btn-sm" onClick={onBack} style={{ marginBottom: 18, paddingLeft: 0 }}>← Back to directory</button>

        {/* header card */}
        <div className="card" style={{ padding: 24, marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
            <div className="avatar" style={{ width: 72, height: 72, background: tint.bg, color: tint.fg, fontSize: 24, borderRadius: 18 }}>{initials(lawyer.name)}</div>
            <div style={{ flex: 1, minWidth: 220 }}>
              <h1 style={{ font: "700 24px var(--font-head)" }}>{lawyer.name}</h1>
              <div style={{ font: "500 13.5px var(--font-body)", color: "var(--muted)", marginTop: 4 }}>{lawyer.specialty}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, flexWrap: "wrap", font: "500 13px var(--font-body)", color: "var(--muted)" }}>
                <span><span className="star">★</span> {lawyer.rating}</span>
                <span style={{ color: "#dcdad4" }}>·</span>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <span className="dot" style={{ background: availColor }} />
                  <span style={{ color: availColor }}>{lawyer.availability}</span>
                </span>
                <span style={{ color: "#dcdad4" }}>·</span>
                <span>{lawyer.location}</span>
              </div>
            </div>
            <div style={{ textAlign: "right", display: "flex", flexDirection: "column", gap: 10, alignItems: "flex-end" }}>
              <div style={{ font: "700 20px var(--font-head)" }}>{formatRate(lawyer.hourlyRate)}</div>
              <button className="btn btn-primary btn-lg" onClick={() => onBook(lawyer)}>Book consultation</button>
            </div>
          </div>
        </div>

        {/* stats */}
        <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 18 }}>
          {stats.map((s) => (
            <div key={s.label} className="card" style={{ padding: "16px 18px" }}>
              <div style={{ font: "600 12px var(--font-body)", color: "var(--muted-3)" }}>{s.label}</div>
              <div style={{ font: "700 22px var(--font-head)", marginTop: 4 }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* bio */}
        <div className="card" style={{ padding: 24, marginBottom: 18 }}>
          <div className="section-label">About</div>
          <p style={{ font: "400 14.5px/1.7 var(--font-body)", color: "var(--text-2)" }}>{lawyer.bio}</p>
        </div>

        {/* details */}
        <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          <div className="card" style={{ padding: 24 }}>
            <div className="section-label">Education</div>
            <p style={{ font: "400 14px/1.6 var(--font-body)", color: "var(--text-2)" }}>{lawyer.education}</p>
          </div>
          <div className="card" style={{ padding: 24 }}>
            <div className="section-label">Languages</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {(lawyer.languages || []).map((l) => <span key={l} className="pill">{l}</span>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
