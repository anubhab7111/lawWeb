import { useState } from "react";
import { register } from "../api";
import { AuthAside } from "./AuthAside";

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
    <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
      <AuthAside
        eyebrow="A first consultation"
        title={<>Understand where<br />you stand.</>}
        body="Create an account to ask questions, review documents, and retain the right advocate — free to begin, no card required."
      />

      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "56px 24px" }}>
        <div className="fade-in" style={{ width: "100%", maxWidth: 380 }}>
          <div style={{ marginBottom: 28 }}>
            <div className="eyebrow" style={{ marginBottom: 12 }}>Create account</div>
            <h1 style={{ font: "600 30px var(--font-head)", letterSpacing: "-.02em" }}>Create your account</h1>
            <p style={{ font: "400 14.5px var(--font-serif)", color: "var(--muted)", marginTop: 8 }}>
              Free to start — no card required.
            </p>
          </div>

          {error && (
            <div style={{ background: "color-mix(in srgb, var(--danger) 12%, transparent)", color: "var(--danger)", borderLeft: "2px solid var(--danger)", borderRadius: "var(--r-sm)", padding: "11px 14px", font: "500 13px var(--font-body)", marginBottom: 18 }}>
              {error}
            </div>
          )}

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
            <button type="submit" className="btn btn-primary btn-block btn-lg" disabled={loading} style={{ marginTop: 8 }}>
              {loading ? "Creating account…" : "Sign up free"}
            </button>
          </form>

          <div style={{ textAlign: "center", marginTop: 22, font: "400 13.5px var(--font-body)", color: "var(--muted-2)" }}>
            Already have an account?{" "}
            <button onClick={onNavigateToSignIn} className="btn btn-ghost btn-sm" style={{ padding: 0 }}>Sign in</button>
          </div>
        </div>
      </div>
    </div>
  );
}
