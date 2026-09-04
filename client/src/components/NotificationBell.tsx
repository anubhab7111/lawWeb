import { useEffect, useRef, useState } from "react";
import { fetchNotifications, markNotificationRead, type AppNotification } from "../api";
import { IconBell } from "./icons";
import type { UserProfile } from "../lib/ui";

interface Props {
  user: UserProfile | null;
}

export function NotificationBell({ user }: Props) {
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!user) return;
    fetchNotifications().then(setNotifications).catch(() => {});
  }, [user]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  if (!user) return null;

  const unread = notifications.filter((n) => !n.readAt).length;

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="theme-toggle"
        style={{ position: "relative" }}
        aria-label="Notifications"
      >
        <IconBell size={17} />
        {unread > 0 && (
          <span style={{
            position: "absolute", top: 5, right: 5, width: 7, height: 7, borderRadius: "50%",
            background: "var(--danger)", border: "1.5px solid var(--surface)",
          }} />
        )}
      </button>
      {open && (
        <div className="modal" style={{ position: "absolute", right: 0, top: 42, width: 300, maxHeight: 360, overflowY: "auto", padding: 6, boxShadow: "var(--shadow-hover)" }}>
          {notifications.length === 0 ? (
            <div style={{ padding: 14, font: "400 12.5px var(--font-body)", color: "var(--muted-2)" }}>No notifications yet.</div>
          ) : (
            notifications.map((n) => (
              <div
                key={n.id}
                className="menu-row"
                style={{ display: "block", textAlign: "left", opacity: n.readAt ? 0.6 : 1 }}
                onClick={() => {
                  if (!n.readAt) {
                    markNotificationRead(n.id).then(() =>
                      setNotifications((prev) => prev.map((x) => (x.id === n.id ? { ...x, readAt: new Date().toISOString() } : x)))
                    );
                  }
                }}
              >
                <div style={{ font: "600 12.5px var(--font-body)" }}>{n.title}</div>
                <div style={{ font: "400 12px var(--font-body)", color: "var(--muted-2)" }}>{n.body}</div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
