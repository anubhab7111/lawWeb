import { useState } from "react";
import { login } from "../api";

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
    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "48px 24px" }}>
      <div className="card" style={{ width: "100%", maxWidth: 400, padding: 32 }}>
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <div style={{ width: 44, height: 44, borderRadius: 14, background: "var(--accent)", color: "#fff", display: "inline-flex", alignItems: "center", justifyContent: "center", font: "700 18px var(--font-head)" }}>L</div>
          <h1 style={{ font: "700 22px var(--font-head)", marginTop: 16 }}>Welcome back</h1>
          <p style={{ font: "400 13.5px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>Sign in to your LawWeb account.</p>
        </div>

        {error && <div style={{ background: "#fbecea", color: "var(--danger)", borderRadius: 10, padding: "10px 14px", font: "500 13px var(--font-body)", marginBottom: 16 }}>{error}</div>}

        <form onSubmit={submit}>
          <div className="field">
            <label>Email</label>
            <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required />
          </div>
          <div className="field">
            <label>Password</label>
            <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required />
          </div>
          <button type="submit" className="btn btn-primary btn-block btn-lg" disabled={loading} style={{ marginTop: 6 }}>
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <div style={{ textAlign: "center", marginTop: 18, font: "400 13.5px var(--font-body)", color: "var(--muted-2)" }}>
          New to LawWeb? <button onClick={onNavigateToSignUp} className="btn btn-ghost btn-sm" style={{ padding: 0 }}>Create an account</button>
        </div>
      </div>
    </div>
  );
}
