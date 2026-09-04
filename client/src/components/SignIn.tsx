import { useState } from "react";
import { login } from "../api";
import { AuthAside } from "./AuthAside";

interface Props {
  onSuccess: (user: any) => void;
  onNavigateToSignUp: () => void;
}

export function SignIn({ onSuccess, onNavigateToSignUp }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await login({ email, password });
      localStorage.setItem("token", data.token);
      onSuccess(data.user);
    } catch (err: any) {
      setError(err.message || "Failed to sign in");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
      <AuthAside
        eyebrow="Welcome back"
        title={<>Your counsel,<br />where you left it.</>}
        body="Sign in to revisit past consultations, saved documents and upcoming appointments."
      />

      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "56px 24px" }}>
        <div className="fade-in" style={{ width: "100%", maxWidth: 380 }}>
          <div style={{ marginBottom: 28 }}>
            <div className="eyebrow" style={{ marginBottom: 12 }}>Sign in</div>
            <h1 style={{ font: "600 30px var(--font-head)", letterSpacing: "-.02em" }}>Welcome back</h1>
            <p style={{ font: "400 14.5px var(--font-serif)", color: "var(--muted)", marginTop: 8 }}>
              Enter your details to continue.
            </p>
          </div>

          {error && (
            <div style={{ background: "color-mix(in srgb, var(--danger) 12%, transparent)", color: "var(--danger)", borderLeft: "2px solid var(--danger)", borderRadius: "var(--r-sm)", padding: "11px 14px", font: "500 13px var(--font-body)", marginBottom: 18 }}>
              {error}
            </div>
          )}

          <form onSubmit={submit}>
            <div className="field">
              <label>Email</label>
              <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required />
            </div>
            <div className="field">
              <label>Password</label>
              <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required />
            </div>
            <button type="submit" className="btn btn-primary btn-block btn-lg" disabled={loading} style={{ marginTop: 8 }}>
              {loading ? "Signing in…" : "Sign in"}
            </button>
          </form>

          <div style={{ textAlign: "center", marginTop: 22, font: "400 13.5px var(--font-body)", color: "var(--muted-2)" }}>
            New to LawWeb?{" "}
            <button onClick={onNavigateToSignUp} className="btn btn-ghost btn-sm" style={{ padding: 0 }}>Create an account</button>
          </div>
        </div>
      </div>
    </div>
  );
}
