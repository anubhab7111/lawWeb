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

// Warm avatar tints matching the wireframe palette; picked deterministically.
const AVATAR_TINTS: { bg: string; fg: string }[] = [
  { bg: "#f1ede4", fg: "#8b7355" },
  { bg: "#e5eef0", fg: "#4a7c85" },
  { bg: "#f1e4ec", fg: "#916080" },
  { bg: "#eaeef1", fg: "#5a6c8a" },
  { bg: "#e9f0e5", fg: "#5f7d4a" },
  { bg: "#f0e9e4", fg: "#8a6a4a" },
];

export function avatarTint(seed: string): { bg: string; fg: string } {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return AVATAR_TINTS[h % AVATAR_TINTS.length];
}

/** Format an integer hourly rate as the wireframe's "₹2,500/hr". */
export function formatRate(rate: number): string {
  return `₹${Number(rate).toLocaleString("en-IN")}/hr`;
}

export function formatMoney(amount: number): string {
  return `₹${Number(amount).toLocaleString("en-IN")}`;
}

// Maps a free-text availability string to a status dot color.
export function availabilityColor(availability: string): string {
  const a = (availability || "").toLowerCase();
  if (a.includes("today") || a.includes("available now") || a.includes("now")) return "#3e9a5f";
  if (a.includes("tomorrow") || a.includes("soon")) return "#c98a1e";
  return "#a3a29c";
}
