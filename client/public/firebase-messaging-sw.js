/* Receives FCM pushes while the site isn't in the foreground.
   The Firebase web config arrives as ?config=<json> when this worker is
   registered (see NotificationSettings.tsx), so no secrets are baked in here. */
importScripts("https://www.gstatic.com/firebasejs/10.14.1/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/10.14.1/firebase-messaging-compat.js");

const params = new URLSearchParams(self.location.search);
const config = JSON.parse(params.get("config") || "null");

if (config) {
  firebase.initializeApp(config);
  const messaging = firebase.messaging();
  messaging.onBackgroundMessage((payload) => {
    const n = payload.notification || {};
    self.registration.showNotification(n.title || "LawWeb", { body: n.body || "" });
  });
}
