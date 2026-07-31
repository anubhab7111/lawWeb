import { useState, useRef, useEffect } from "react";
import type { View } from "../App";
import { initials, type UserProfile } from "../lib/ui";

interface NavProps {
  view: View;
  user: UserProfile | null;
  marketing?: boolean;
  onNavigate: (v: View) => void;
  onLogout: () => void;
}

const TABS: { key: View; label: string; auth?: boolean }[] = [
  { key: "chat", label: "Ask AI" },
  { key: "lawyers", label: "Find Lawyers" },
  { key: "bookings", label: "My Bookings", auth: true },
];

export function Nav({ view, user, marketing, onNavigate, onLogout }: NavProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  return (
    <div className="nav">
      <div className="brand" onClick={() => onNavigate("home")}>
        <div className="brand-logo">L</div>
        <span className="brand-name">LawWeb</span>
      </div>

      {!marketing && (
        <div className="nav-tabs">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`nav-tab${view === t.key ? " active" : ""}`}
              onClick={() => onNavigate(t.auth && !user ? "signin" : t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      <div className="nav-right">
        {user ? (
          <div ref={menuRef} style={{ position: "relative" }}>
            <button
              onClick={() => setMenuOpen((o) => !o)}
              style={{
                display: "flex", alignItems: "center", gap: 10, background: "none",
                border: "none", cursor: "pointer", padding: 0,
              }}
            >
              <span className="nav-user">{user.name}</span>
              <span
                className="avatar avatar-round"
                style={{ width: 30, height: 30, background: "#e4e2dc", color: "#5c6068", fontSize: 12 }}
              >
                {initials(user.name)}
              </span>
            </button>
            {menuOpen && (
              <div
                className="modal"
                style={{ position: "absolute", right: 0, top: 42, width: 180, padding: 6, boxShadow: "var(--shadow-hover)" }}
              >
                <button className="menu-row" onClick={() => { setMenuOpen(false); onNavigate("bookings"); }}>My Bookings</button>
                <button className="menu-row" onClick={() => { setMenuOpen(false); onNavigate("documents"); }}>Analyze a Document</button>
                <div className="divider" style={{ margin: "6px 0" }} />
                <button className="menu-row danger" onClick={() => { setMenuOpen(false); onLogout(); }}>Sign out</button>
              </div>
            )}
          </div>
        ) : (
          <>
            <button className="nav-tab" onClick={() => onNavigate("signin")}>Log in</button>
            <button className="btn btn-primary btn-sm" onClick={() => onNavigate("signup")}>Sign up free</button>
          </>
        )}
      </div>
    </div>
  );
}
