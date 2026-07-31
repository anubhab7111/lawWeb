import { useState, useEffect, useCallback } from "react";
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
import { fetchUserProfile } from "./api";
import type { Lawyer, UserProfile } from "./lib/ui";

export type View =
  | "home" | "chat" | "lawyers" | "profile" | "payment"
  | "bookings" | "documents" | "signin" | "signup";

const AUTH_REQUIRED: View[] = ["bookings", "payment", "profile"];

export default function App() {
  const [view, setView] = useState<View>("home");
  const [user, setUser] = useState<UserProfile | null>(null);
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // Restore session from a stored token.
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;
    fetchUserProfile()
      .then((u) => setUser(u))
      .catch(() => localStorage.removeItem("token"));
  }, []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  }, []);

  const navigate = useCallback((v: View) => {
    if (AUTH_REQUIRED.includes(v) && !localStorage.getItem("token")) {
      setView("signin");
      return;
    }
    setView(v);
  }, []);

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
    if (!user) { setView("signin"); return; }
    setView("payment");
  };

  const handleAskFromHero = (q: string) => {
    setPendingQuestion(q);
    setView("chat");
  };

  const marketing = view === "home" && !user;

  return (
    <div className="app">
      <Nav view={view} user={user} marketing={marketing} onNavigate={navigate} onLogout={handleLogout} />

      {view === "home" && <Home onNavigate={navigate} onAsk={handleAskFromHero} />}

      {view === "chat" && (
        <AskAI
          user={user}
          initialQuestion={pendingQuestion}
          onConsumeInitial={() => setPendingQuestion(null)}
          onBookLawyer={handleBook}
          onViewLawyer={handleSelectLawyer}
          onNavigate={navigate}
        />
      )}

      {view === "lawyers" && (
        <FindLawyers onSelectLawyer={handleSelectLawyer} onBook={handleBook} />
      )}

      {view === "profile" && selectedLawyer && (
        <LawyerProfile lawyer={selectedLawyer} onBook={handleBook} onBack={() => setView("lawyers")} />
      )}

      {view === "payment" && selectedLawyer && (
        <Payment
          lawyer={selectedLawyer}
          user={user}
          onBack={() => setView("profile")}
          onSuccess={() => { showToast("Booking confirmed"); setView("bookings"); }}
        />
      )}

      {view === "bookings" && <MyBookings user={user} onNavigate={navigate} />}

      {view === "documents" && <DocumentAnalysis />}

      {view === "signin" && (
        <SignIn onSuccess={handleLoginSuccess} onNavigateToSignUp={() => setView("signup")} />
      )}
      {view === "signup" && (
        <SignUp onSuccess={handleLoginSuccess} onNavigateToSignIn={() => setView("signin")} />
      )}

      {toast && (
        <div className="toast-wrap"><div className="toast">✓ {toast}</div></div>
      )}
    </div>
  );
}
