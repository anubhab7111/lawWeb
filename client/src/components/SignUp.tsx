import { useState } from "react";
import { register } from "../api";

interface Props {
  onSuccess: (user: any) => void;
  onNavigateToSignIn: () => void;
}

export function SignUp({ onSuccess, onNavigateToSignIn }: Props) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await register({ name, email, password });
      localStorage.setItem("token", data.token);
      onSuccess(data.user);
    } catch (err: any) {
      setError(err.message || "Failed to create account");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "48px 24px" }}>
      <div className="card" style={{ width: "100%", maxWidth: 400, padding: 32 }}>
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <div style={{ width: 44, height: 44, borderRadius: 14, background: "var(--accent)", color: "#fff", display: "inline-flex", alignItems: "center", justifyContent: "center", font: "700 18px var(--font-head)" }}>L</div>
          <h1 style={{ font: "700 22px var(--font-head)", marginTop: 16 }}>Create your account</h1>
          <p style={{ font: "400 13.5px var(--font-body)", color: "var(--muted-2)", marginTop: 4 }}>Free to start — no card required.</p>
        </div>

        {error && <div style={{ background: "#fbecea", color: "var(--danger)", borderRadius: 10, padding: "10px 14px", font: "500 13px var(--font-body)", marginBottom: 16 }}>{error}</div>}

        <form onSubmit={submit}>
          <div className="field">
            <label>Full name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Asha Rao" required />
          </div>
          <div className="field">
            <label>Email</label>
            <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required />
          </div>
          <div className="field">
            <label>Password</label>
            <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 6 characters" required />
          </div>
          <button type="submit" className="btn btn-primary btn-block btn-lg" disabled={loading} style={{ marginTop: 6 }}>
            {loading ? "Creating account…" : "Sign up free"}
          </button>
        </form>

        <div style={{ textAlign: "center", marginTop: 18, font: "400 13.5px var(--font-body)", color: "var(--muted-2)" }}>
          Already have an account? <button onClick={onNavigateToSignIn} className="btn btn-ghost btn-sm" style={{ padding: 0 }}>Sign in</button>
        </div>
      </div>
    </div>
  );
}
