import { useState, useEffect, useRef } from "react";
import DropIn from "braintree-web-drop-in-react";
import { fetchBraintreeClientToken, checkoutBooking } from "../api";
import { IconLock } from "./icons";
import { initials, avatarTint, formatMoney, currentCurrency, type Lawyer, type UserProfile } from "../lib/ui";

const TIME_SLOTS = Array.from({ length: 19 }, (_, i) => {
  const minutes = 9 * 60 + i * 30;
  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
});

function newKey(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function localToday(): string {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 10);
}

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
  const [date, setDate] = useState(localToday());
  const [time, setTime] = useState("10:00");
  const [unrecorded, setUnrecorded] = useState<string | null>(null);
  // One key per payment attempt: a retry after a dropped connection reuses it
  // (so the server can't charge twice); a declined attempt gets a fresh one.
  const idempotencyKey = useRef(newKey());

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
    if (!user) {
      setError("Your session has expired. Please sign in again to complete this booking.");
      return;
    }
    if (!instance) {
      setError("The payment form is still loading — please wait a moment and try again.");
      return;
    }
    if (!date || !time) {
      setError("Choose an appointment date and time.");
      return;
    }
    setProcessing(true);
    setError(null);
    try {
      const { nonce } = await instance.requestPaymentMethod();
      const result = await checkoutBooking(
        {
          amount: total.toFixed(2),
          paymentMethodNonce: nonce,
          lawyerId: lawyer.id,
          userId: user.id,
          appointmentDate: date,
          appointmentTime: time,
        },
        idempotencyKey.current,
      );
      if (result.warning) {
        setUnrecorded(result.transactionId);
        return;
      }
      onSuccess();
    } catch (e: any) {
      if (!(e instanceof TypeError)) idempotencyKey.current = newKey();
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
          <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
            <div className="field" style={{ flex: 1, minWidth: 150, marginBottom: 0 }}>
              <label>Appointment date</label>
              <input className="input" type="date" min={localToday()} value={date} onChange={(e) => setDate(e.target.value)} />
            </div>
            <div className="field" style={{ flex: 1, minWidth: 120, marginBottom: 0 }}>
              <label>Time (IST)</label>
              <select className="input" value={time} onChange={(e) => setTime(e.target.value)}>
                {TIME_SLOTS.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>
          {[
            { label: "Consultation", value: formatMoney(consultation) },
            { label: "Platform fee (5%)", value: formatMoney(fee) },
          ].map((r) => (
            <div key={r.label} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", font: "400 13.5px var(--font-body)", color: "var(--muted)" }}>
              <span>{r.label}</span><span className="mono-num">{r.value}</span>
            </div>
          ))}
          <div className="divider" style={{ margin: "10px 0" }} />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ font: "600 14px var(--font-body)" }}>Total</span>
            <span className="mono-num" style={{ font: "700 22px var(--font-head)", color: "var(--accent)" }}>{formatMoney(total)}</span>
          </div>
        </div>

        {/* payment */}
        <div className="card" style={{ padding: 22 }}>
          <div style={{ font: "700 16px var(--font-head)", marginBottom: 4 }}>Secure payment</div>
          <div style={{ font: "400 13px var(--font-body)", color: "var(--muted-2)", marginBottom: 18 }}>Card details are processed by Braintree (sandbox). Use a test card to complete a booking.</div>

          <div style={{ font: "400 12px var(--font-body)", color: "var(--muted-3)", marginBottom: 12 }}>
            Charged in {currentCurrency()}.
          </div>

          {unrecorded && (
            <div className="error-banner" style={{ marginBottom: 14 }}>
              Your payment went through, but we couldn't save the booking. Please contact support with
              transaction ID <strong>{unrecorded}</strong> — do not pay again.
            </div>
          )}

          {error && <div className="error-banner" style={{ marginBottom: 14 }}>{error}</div>}

          {clientToken ? (
            <>
              <DropIn
                options={{ authorization: clientToken, card: { cardholderName: { required: true } } }}
                onInstance={(inst: any) => setInstance(inst)}
              />
              <button className="btn btn-primary btn-block btn-lg" disabled={processing || !!unrecorded} onClick={pay} style={{ marginTop: 12 }}>
                {processing ? "Processing…" : `Pay ${formatMoney(total)}`}
              </button>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 12, font: "500 11px var(--font-body)", letterSpacing: ".08em", textTransform: "uppercase", color: "var(--muted-3)" }}><IconLock size={13} /> PCI-DSS compliant gateway</div>
            </>
          ) : !error ? (
            <div style={{ padding: "48px 0", display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
              <div className="spinner" style={{ width: 28, height: 28, borderColor: "var(--border-2)", borderTopColor: "var(--accent)" }} />
              <div style={{ font: "500 13px var(--font-body)", color: "var(--muted-2)" }}>Establishing a secure connection…</div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
