import { useState, useEffect } from "react";
import DropIn from "braintree-web-drop-in-react";
import { fetchBraintreeClientToken, checkoutBooking } from "../api";
import { initials, avatarTint, formatMoney, type Lawyer, type UserProfile } from "../lib/ui";

interface Props {
  lawyer: Lawyer;
  user: UserProfile | null;
  onBack: () => void;
  onSuccess: () => void;
}

export function Payment({ lawyer, user, onBack, onSuccess }: Props) {
  const [clientToken, setClientToken] = useState<string | null>(null);
  const [instance, setInstance] = useState<any>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const consultation = lawyer.hourlyRate;
  const fee = Math.round(consultation * 0.05);
  const total = consultation + fee;
  const tint = avatarTint(lawyer.name);

  useEffect(() => {
    fetchBraintreeClientToken()
      .then(setClientToken)
      .catch(() => setError("Couldn't reach the payment gateway. Check the backend's Braintree keys."));
  }, []);

  const pay = async () => {
    if (!instance || !user) return;
    setProcessing(true);
    setError(null);
    try {
      const { nonce } = await instance.requestPaymentMethod();
      await checkoutBooking({
        amount: total.toFixed(2),
        paymentMethodNonce: nonce,
        lawyerId: lawyer.id,
        userId: user.id,
      });
      onSuccess();
    } catch (e: any) {
      setError(e.message || "Payment failed. Please try again.");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div className="container" style={{ maxWidth: 620 }}>
        <button className="btn btn-ghost btn-sm" onClick={onBack} style={{ marginBottom: 18, paddingLeft: 0 }}>← Back to profile</button>

        {/* booking summary */}
        <div className="card" style={{ padding: 22, marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 14, alignItems: "center", marginBottom: 18 }}>
            <div className="avatar" style={{ width: 48, height: 48, background: tint.bg, color: tint.fg }}>{initials(lawyer.name)}</div>
            <div style={{ flex: 1 }}>
              <div style={{ font: "700 15px var(--font-head)" }}>{lawyer.name}</div>
              <div style={{ font: "500 12.5px var(--font-body)", color: "var(--muted-2)" }}>{lawyer.specialty} · Consultation</div>
            </div>
          </div>
          {[
            { label: "Consultation", value: formatMoney(consultation) },
            { label: "Platform fee (5%)", value: formatMoney(fee) },
          ].map((r) => (
            <div key={r.label} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", font: "400 13.5px var(--font-body)", color: "var(--muted)" }}>
              <span>{r.label}</span><span>{r.value}</span>
            </div>
          ))}
          <div className="divider" style={{ margin: "10px 0" }} />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ font: "600 14px var(--font-body)" }}>Total</span>
            <span style={{ font: "700 22px var(--font-head)", color: "var(--accent)" }}>{formatMoney(total)}</span>
          </div>
        </div>

        {/* payment */}
        <div className="card" style={{ padding: 22 }}>
          <div style={{ font: "700 16px var(--font-head)", marginBottom: 4 }}>Secure payment</div>
          <div style={{ font: "400 13px var(--font-body)", color: "var(--muted-2)", marginBottom: 18 }}>Card details are processed by Braintree (sandbox). Use a test card to complete a booking.</div>

          {error && <div style={{ background: "#fbecea", color: "var(--danger)", borderRadius: 10, padding: "10px 14px", font: "500 13px var(--font-body)", marginBottom: 14 }}>{error}</div>}

          {clientToken ? (
            <>
              <DropIn
                options={{ authorization: clientToken, card: { cardholderName: { required: true } } }}
                onInstance={(inst: any) => setInstance(inst)}
              />
              <button className="btn btn-primary btn-block btn-lg" disabled={processing} onClick={pay} style={{ marginTop: 12 }}>
                {processing ? "Processing…" : `Pay ${formatMoney(total)}`}
              </button>
              <div style={{ textAlign: "center", marginTop: 12, font: "500 11px var(--font-body)", letterSpacing: ".08em", textTransform: "uppercase", color: "var(--muted-3)" }}>🔒 PCI-DSS compliant gateway</div>
            </>
          ) : !error ? (
            <div style={{ padding: "48px 0", display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
              <div className="spinner" style={{ width: 28, height: 28, borderColor: "rgba(44,110,107,.25)", borderTopColor: "var(--accent)" }} />
              <div style={{ font: "500 13px var(--font-body)", color: "var(--muted-2)" }}>Establishing a secure connection…</div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
