import { useEffect, useState } from "react";
import { fetchAppConfig, fetchNotificationPrefs, updateNotificationPrefs, type NotificationPrefs } from "../api";

const FIREBASE_SDK = "https://www.gstatic.com/firebasejs/10.14.1";

// The Firebase SDK is loaded from Google's CDN on demand (only when the user
// opts in to browser push), so it isn't a build-time dependency.
async function requestPushToken(config: Record<string, unknown>, vapidKey: string): Promise<string> {
  if (!("serviceWorker" in navigator) || !("Notification" in window)) {
    throw new Error("This browser doesn't support push notifications.");
  }
  const permission = await Notification.requestPermission();
  if (permission !== "granted") throw new Error("Notification permission was not granted.");

  const registration = await navigator.serviceWorker.register(
    `/firebase-messaging-sw.js?config=${encodeURIComponent(JSON.stringify(config))}`,
  );
  const { initializeApp } = await import(/* @vite-ignore */ `${FIREBASE_SDK}/firebase-app.js`);
  const { getMessaging, getToken } = await import(/* @vite-ignore */ `${FIREBASE_SDK}/firebase-messaging.js`);
  const messaging = getMessaging(initializeApp(config));
  const token = await getToken(messaging, { vapidKey, serviceWorkerRegistration: registration });
  if (!token) throw new Error("Couldn't get a push token from Firebase.");
  return token;
}

export function NotificationSettings() {
  const [prefs, setPrefs] = useState<NotificationPrefs | null>(null);
  const [firebase, setFirebase] = useState<{ config: Record<string, unknown>; vapidKey: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchNotificationPrefs().then(setPrefs).catch(() => setError("Couldn't load your notification settings."));
    fetchAppConfig().then((c) => setFirebase(c.firebase ?? null)).catch(() => {});
  }, []);

  const save = async (patch: Partial<NotificationPrefs> & { fcmToken?: string }) => {
    if (!prefs) return;
    setBusy(true);
    setError(null);
    try {
      setPrefs(await updateNotificationPrefs({ emailEnabled: prefs.emailEnabled, pushEnabled: prefs.pushEnabled, ...patch }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't save that change.");
    } finally {
      setBusy(false);
    }
  };

  const enablePush = async () => {
    if (!firebase) return;
    setBusy(true);
    setError(null);
    try {
      const fcmToken = await requestPushToken(firebase.config, firebase.vapidKey);
      setPrefs(await updateNotificationPrefs({ emailEnabled: prefs?.emailEnabled ?? true, pushEnabled: true, fcmToken }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't enable push notifications.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 620 }}>
        <h1 className="page-title">Notification settings</h1>
        <p className="page-sub">Choose how LawWeb reaches you about hearings, orders and shared documents.</p>

        {error && <div className="error-banner" style={{ marginBottom: 18 }}>{error}</div>}

        {!prefs ? (
          <div className="shimmer" style={{ height: 120, borderRadius: "var(--r-lg)" }} />
        ) : (
          <div className="card" style={{ padding: 22, display: "flex", flexDirection: "column", gap: 18 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 10, font: "500 14px var(--font-body)" }}>
              <input type="checkbox" checked disabled /> In-app notifications (always on)
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 10, font: "500 14px var(--font-body)" }}>
              <input type="checkbox" checked={prefs.emailEnabled} disabled={busy} onChange={(e) => save({ emailEnabled: e.target.checked })} />
              Email
            </label>

            {firebase ? (
              <div>
                <div style={{ font: "500 14px var(--font-body)", marginBottom: 6 }}>Browser push notifications</div>
                {prefs.pushEnabled && prefs.hasFcmToken ? (
                  <button className="btn btn-outline btn-sm" disabled={busy} onClick={() => save({ pushEnabled: false, fcmToken: "" })}>
                    Turn off push on this account
                  </button>
                ) : (
                  <button className="btn btn-primary btn-sm" disabled={busy} onClick={enablePush}>
                    Enable push on this browser
                  </button>
                )}
              </div>
            ) : (
              <div style={{ font: "400 13px var(--font-body)", color: "var(--muted-2)" }}>
                Browser push isn't set up on this server yet.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
