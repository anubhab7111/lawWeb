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
    <div className="card card-hover" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
        <div className="avatar" style={{ width: 48, height: 48, background: tint.bg, color: tint.fg, fontSize: 16 }}>{initials(lawyer.name)}</div>
        <div style={{ minWidth: 0 }}>
          <div style={{ font: "600 17px var(--font-head)", letterSpacing: "-.01em" }}>{lawyer.name}</div>
          <div style={{ font: "500 12px var(--font-body)", letterSpacing: ".01em", color: "var(--muted-2)", marginTop: 2 }}>
            {lawyer.specialty} · {lawyer.experience} yrs
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4, font: "600 13px var(--font-body)", color: "var(--text-2)" }}>
          <span className="star">★</span><span className="mono-num">{lawyer.rating}</span>
        </div>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, font: "500 12.5px var(--font-body)", color: "var(--muted)" }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span className="dot" style={{ background: availColor }} />
          <span style={{ color: availColor }}>{lawyer.availability}</span>
        </span>
        <span style={{ color: "var(--faint)" }}>·</span>
        <span>{lawyer.location}</span>
        <span style={{ color: "var(--faint)" }}>·</span>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 160 }}>{(lawyer.languages || []).join(", ")}</span>
      </div>

      <div className="divider" />

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div>
          <div className="mono-num" style={{ font: "600 18px var(--font-head)", letterSpacing: "-.01em" }}>{formatRate(lawyer.hourlyRate)}</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <button className="btn btn-outline btn-sm" onClick={() => onView(lawyer)}>Profile</button>
          <button className="btn btn-primary btn-sm" onClick={() => onBook(lawyer)}>Book</button>
        </div>
      </div>
    </div>
  );
}
