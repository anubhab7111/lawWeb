// Small presentation helpers shared across LawWeb screens.

export interface Lawyer {
  id: string;
  name: string;
  specialty: string;
  experience: number;
  rating: number;
  hourlyRate: number;
  location: string;
  bio: string;
  cases: number;
  successRate: number;
  education: string;
  languages: string[];
  availability: string;
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
}

/** "Rhea Mehta" -> "RM" */
export function initials(name: string): string {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  return ((parts[0]?.[0] ?? "") + (parts[1]?.[0] ?? "")).toUpperCase() || "?";
}

// Muted, in-palette avatar tints; picked deterministically per name. The
// values are CSS custom properties (defined for both themes in theme.css) so
// avatars stay legible on the parchment and the dark grounds alike.
const AVATAR_TINTS: { bg: string; fg: string }[] = [
  { bg: "var(--av1-bg)", fg: "var(--av1-fg)" },
  { bg: "var(--av2-bg)", fg: "var(--av2-fg)" },
  { bg: "var(--av3-bg)", fg: "var(--av3-fg)" },
  { bg: "var(--av4-bg)", fg: "var(--av4-fg)" },
  { bg: "var(--av5-bg)", fg: "var(--av5-fg)" },
  { bg: "var(--av6-bg)", fg: "var(--av6-fg)" },
];

export function avatarTint(seed: string): { bg: string; fg: string } {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return AVATAR_TINTS[h % AVATAR_TINTS.length];
}

// The currency Braintree actually charges (from /api/config); prices are shown
// in it rather than assuming rupees.
let currencyCode = "USD";

export function setCurrency(code: string) {
  if (code) currencyCode = code.toUpperCase();
}

export function currentCurrency(): string {
  return currencyCode;
}

function money(amount: number): string {
  try {
    return new Intl.NumberFormat("en-IN", { style: "currency", currency: currencyCode, maximumFractionDigits: 0 }).format(Number(amount));
  } catch {
    return `${currencyCode} ${Number(amount).toLocaleString("en-IN")}`;
  }
}

export function formatRate(rate: number): string {
  return `${money(rate)}/hr`;
}

export function formatMoney(amount: number): string {
  return money(amount);
}

// Maps a free-text availability string to a status dot color (theme tokens).
export function availabilityColor(availability: string): string {
  const a = (availability || "").toLowerCase();
  if (a.includes("today") || a.includes("available now") || a.includes("now")) return "var(--green)";
  if (a.includes("tomorrow") || a.includes("soon")) return "var(--amber)";
  return "var(--muted-3)";
}
