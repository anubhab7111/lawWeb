import { initials, avatarTint, availabilityColor, formatRate, type Lawyer } from "../lib/ui";

interface Props {
  lawyer: Lawyer;
  onView: (l: Lawyer) => void;
  onBook: (l: Lawyer) => void;
}

export function LawyerCard({ lawyer, onView, onBook }: Props) {
  const tint = avatarTint(lawyer.name);
  const availColor = availabilityColor(lawyer.availability);
  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <div className="avatar" style={{ width: 46, height: 46, background: tint.bg, color: tint.fg, fontSize: 15 }}>{initials(lawyer.name)}</div>
        <div>
          <div style={{ font: "700 14.5px var(--font-head)" }}>{lawyer.name}</div>
          <div style={{ font: "500 12.5px var(--font-body)", color: "var(--muted-2)" }}>{lawyer.specialty} · {lawyer.experience} yrs</div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, font: "500 12.5px var(--font-body)", color: "var(--muted)" }}>
        <span className="star">★</span><span>{lawyer.rating}</span>
        <span style={{ color: "#dcdad4" }}>·</span>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{(lawyer.languages || []).join(", ")}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span className="dot" style={{ background: availColor }} />
        <span style={{ font: "500 12px var(--font-body)", color: availColor }}>{lawyer.availability}</span>
        <span style={{ color: "#dcdad4" }}>·</span>
        <span style={{ font: "500 12px var(--font-body)", color: "var(--muted-2)" }}>{lawyer.location}</span>
      </div>
      <div style={{ font: "600 13.5px var(--font-body)", marginTop: 2 }}>{formatRate(lawyer.hourlyRate)}</div>
      <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
        <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={() => onBook(lawyer)}>Book</button>
        <button className="btn btn-outline btn-sm" style={{ flex: 1 }} onClick={() => onView(lawyer)}>View profile</button>
      </div>
    </div>
  );
}
