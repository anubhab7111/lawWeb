import { useState, useEffect, useCallback } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Nav } from "./components/Nav";
import { Home } from "./components/Home";
import { AskAI } from "./components/AskAI";
import { FindLawyers } from "./components/FindLawyers";
import { LawyerProfile } from "./components/LawyerProfile";
import { Payment } from "./components/Payment";
import { MyBookings } from "./components/MyBookings";
import { DocumentAnalysis } from "./components/DocumentAnalysis";
import { SignIn } from "./components/SignIn";
import { SignUp } from "./components/SignUp";
import { BareActExplorer } from "./components/BareActExplorer";
import { SimilarCaseSearch } from "./components/SimilarCaseSearch";
import { MyCases } from "./components/MyCases";
import { CauseListSearch } from "./components/CauseListSearch";
import { Vault } from "./components/Vault";
import { LegalCalendar } from "./components/LegalCalendar";
import { IconCheck } from "./components/icons";
import { fetchUserProfile } from "./api";
import type { Lawyer, UserProfile } from "./lib/ui";

export type View =
  | "home" | "chat" | "lawyers" | "profile" | "payment"
  | "bookings" | "documents" | "signin" | "signup"
  | "bare-acts" | "similar-cases" | "my-cases" | "cause-list" | "vault" | "calendar";

const ALL_VIEWS: View[] = [
  "home", "chat", "lawyers", "profile", "payment",
  "bookings", "documents", "signin", "signup",
  "bare-acts", "similar-cases", "my-cases", "cause-list", "vault", "calendar",
];

// Views that render around an in-memory object (the selected lawyer) rather
// than anything fetchable from a URL — a cold deep-link or a Back/Forward
// landing here with nothing selected has no data to show.
const NEEDS_SELECTED_LAWYER: View[] = ["profile", "payment"];

function hashToView(hash: string): View | null {
  const v = hash.replace(/^#\/?/, "");
  return (ALL_VIEWS as string[]).includes(v) ? (v as View) : null;
}

// "profile" (a lawyer's public profile) is deliberately not gated — it's
// browsable by guests; only booking requires auth. It was previously listed
// here but unreachable via navigate() anyway (handleSelectLawyer sets the
// view directly), so the entry was inert rather than intentional.
const AUTH_REQUIRED: View[] = ["bookings", "payment", "my-cases", "vault", "calendar"];

export default function App() {
  const [view, setView] = useState<View>(() => hashToView(window.location.hash) ?? "home");
  const [user, setUser] = useState<UserProfile | null>(null);
  // Distinguishes "haven't checked the stored token yet" from "checked, and
  // there's no user" — components gated on `user` (MyBookings, Payment) used
  // to hang or silently no-op during the brief window after mount where a
  // valid token exists but the /auth/me fetch hasn't resolved, because
  // navigate() gated on localStorage's token while those components gated
  // on `user`. Gating navigate() on this instead closes that race instead
  // of just papering over it in each component.
  const [authChecked, setAuthChecked] = useState(false);
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // What actually renders: falls back to "lawyers" for profile/payment
  // reached with no lawyer in memory — a cold deep-link, a Back/Forward
  // navigation, or a manually-edited URL, none of which carry the selected
  // lawyer object. Without this the app used to render just the nav with
  // nothing below it (App.tsx previously had no fallback branch at all).
  const effectiveView: View =
    NEEDS_SELECTED_LAWYER.includes(view) && !selectedLawyer ? "lawyers" : view;

  // Browser Back/Forward and manually-edited URLs.
  useEffect(() => {
    const onHashChange = () => setView(hashToView(window.location.hash) ?? "home");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Keep the URL in sync with what's actually on screen (effectiveView, not
  // view) so Back/Forward and a page refresh land somewhere real instead of
  // replaying a stale profile/payment hash with nothing to show.
  useEffect(() => {
    const target = `#/${effectiveView}`;
    if (window.location.hash !== target) window.location.hash = target;
  }, [effectiveView]);

  // navigate() gates click-driven navigation, but a cold deep-link (e.g.
  // opening #/bookings directly) sets `view` from the URL without ever
  // going through navigate() — gate those once the stored token has been
  // checked, same rule, so a guest can't land straight on a protected view.
  useEffect(() => {
    if (authChecked && AUTH_REQUIRED.includes(view) && !user) {
      setView("signin");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked]);

  // Restore session from a stored token.
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setAuthChecked(true);
      return;
    }
    fetchUserProfile()
      .then((u) => setUser(u))
      .catch(() => localStorage.removeItem("token"))
      .finally(() => setAuthChecked(true));
  }, []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  }, []);

  const navigate = useCallback((v: View) => {
    if (AUTH_REQUIRED.includes(v)) {
      if (!authChecked) return; // still resolving the stored token — ignore the click rather than bounce to signin
      if (!user) { setView("signin"); return; }
    }
    setView(v);
  }, [authChecked, user]);

  const handleLoginSuccess = (u: UserProfile) => {
    setUser(u);
    setView("home");
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setUser(null);
    setView("home");
  };

  const handleSelectLawyer = (l: Lawyer) => {
    setSelectedLawyer(l);
    setView("profile");
  };

  const handleBook = (l: Lawyer) => {
    setSelectedLawyer(l);
    if (!authChecked) return; // still resolving the stored token
    if (!user) { setView("signin"); return; }
    setView("payment");
  };

  const handleAskFromHero = (q: string) => {
    setPendingQuestion(q);
    setView("chat");
  };

  const marketing = effectiveView === "home" && !user;

  return (
    <div className="app">
      <Nav view={effectiveView} user={user} marketing={marketing} onNavigate={navigate} onLogout={handleLogout} />

      {/* key resets the boundary's error state on navigation, so a crash on
          one screen doesn't stay stuck once the user moves on. */}
      <ErrorBoundary key={effectiveView} onReset={() => setView("home")}>

      {effectiveView === "home" && <Home onNavigate={navigate} onAsk={handleAskFromHero} />}

      {effectiveView === "chat" && (
        <AskAI
          user={user}
          initialQuestion={pendingQuestion}
          onConsumeInitial={() => setPendingQuestion(null)}
          onBookLawyer={handleBook}
          onViewLawyer={handleSelectLawyer}
          onNavigate={navigate}
        />
      )}

      {effectiveView === "lawyers" && (
        <FindLawyers onSelectLawyer={handleSelectLawyer} onBook={handleBook} />
      )}

      {effectiveView === "profile" && selectedLawyer && (
        <LawyerProfile lawyer={selectedLawyer} onBook={handleBook} onBack={() => setView("lawyers")} />
      )}

      {effectiveView === "payment" && selectedLawyer && (
        <Payment
          lawyer={selectedLawyer}
          user={user}
          onBack={() => setView("profile")}
          onSuccess={() => { showToast("Booking confirmed"); setView("bookings"); }}
        />
      )}

      {effectiveView === "bookings" && <MyBookings user={user} onNavigate={navigate} />}

      {effectiveView === "documents" && <DocumentAnalysis />}

      {effectiveView === "bare-acts" && <BareActExplorer />}

      {effectiveView === "similar-cases" && <SimilarCaseSearch />}

      {effectiveView === "my-cases" && <MyCases user={user} />}

      {effectiveView === "cause-list" && <CauseListSearch />}

      {effectiveView === "vault" && <Vault user={user} />}

      {effectiveView === "calendar" && <LegalCalendar user={user} />}

      {effectiveView === "signin" && (
        <SignIn onSuccess={handleLoginSuccess} onNavigateToSignUp={() => setView("signup")} />
      )}
      {effectiveView === "signup" && (
        <SignUp onSuccess={handleLoginSuccess} onNavigateToSignIn={() => setView("signin")} />
      )}

      </ErrorBoundary>

      {toast && (
        <div className="toast-wrap"><div className="toast" style={{ display: "flex", alignItems: "center", gap: 8 }}><IconCheck size={14} /> {toast}</div></div>
      )}
    </div>
  );
}
