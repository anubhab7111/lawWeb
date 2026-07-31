# LawWeb — Frontend Design Concept

Design brief for building the LawWeb client. **Modern & minimal**, light + dark.
This document drives the UI; feature behavior is fixed by the FastAPI backend (`/api/*`).

---

## 1. Product in one line

An AI legal assistant for Indian law — ask questions, analyze documents, get crime-reporting guidance, and find & book lawyers — all backed by local, retrieval-grounded answers.

## 2. Design principles

1. **Clarity over decoration.** Legal content is dense; the UI stays calm and gives it room.
2. **One accent, lots of whitespace.** Neutral canvas, a single blue accent for actions and state.
3. **Trust through transparency.** Always show sources/citations, confidence, and "not legal advice" context so users can judge answers.
4. **Answer-first.** The chatbot is the hero. Every screen is reachable in ≤2 clicks.
5. **Responsive & accessible by default.** Mobile-first layouts, WCAG AA contrast, full keyboard support.

## 3. Brand

- **Name:** LawWeb. **Wordmark:** "LawWeb" in the heading typeface, paired with a simple scale/balance mark.
- **Personality:** precise, quietly confident, approachable — a modern tool, not a law firm.
- **Tagline (optional):** "Understand your legal position."

---

## 4. Design tokens

### Color

Neutral gray canvas + single blue accent. Semantic colors for status only.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#FFFFFF` | `#0B0F14` | Page background |
| `--surface` | `#F7F8FA` | `#141A21` | Cards, panels |
| `--surface-2` | `#EEF1F5` | `#1C232C` | Nested / hover surfaces |
| `--border` | `#E3E8EF` | `#273039` | Hairlines, dividers |
| `--text` | `#0F172A` | `#E7ECF3` | Primary text |
| `--text-muted` | `#5B6472` | `#9AA6B2` | Secondary text, labels |
| `--accent` | `#2563EB` | `#3B82F6` | Primary actions, links, active nav |
| `--accent-weak` | `#EFF4FF` | `#16233B` | Accent backgrounds/badges |
| `--success` | `#16A34A` | `#22C55E` | Confirmed bookings, verified citations |
| `--warning` | `#D97706` | `#F59E0B` | Low-confidence answers, defect flags |
| `--danger` | `#DC2626` | `#EF4444` | Errors, destructive actions |

Theme via `:root` + `[data-theme="dark"]` custom properties; respect `prefers-color-scheme` on first load.

### Typography

- **Type family:** `Inter` (or system sans) for UI/body; a single geometric sans is enough — no serif. Optional `ui-monospace` for statute references and citations.
- **Scale (rem):** display 2.25 / h1 1.75 / h2 1.375 / h3 1.125 / body 1.0 / small 0.875 / caption 0.75.
- **Weights:** 400 body, 500 labels/nav, 600 headings. Line-height 1.5 body, 1.25 headings.
- **Reading width:** cap long text (chat answers, doc analysis) at ~68ch.

### Space, radius, elevation

- **Spacing scale (px):** 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64.
- **Radius:** `sm 6` · `md 10` · `lg 16` · `full` (pills/avatars). Cards use `lg`.
- **Elevation:** flat by default; one soft shadow for raised cards/popovers (`0 1px 2px`, `0 8px 24px` on overlays). Prefer borders over shadows in dark mode.
- **Motion:** 150–200ms ease-out for hovers/entrances; respect `prefers-reduced-motion`. Chat tokens stream in without layout jumps.

---

## 5. Layout & navigation

- **App shell:** sticky top bar (wordmark left; primary nav center/left; auth + theme toggle right). Content max-width `1200px`, centered, `24px` gutters.
- **Primary nav (authed):** Home · Ask AI · Find Lawyers · My Bookings · account menu.
- **Mobile:** top bar collapses to wordmark + hamburger → slide-in sheet; a bottom tab bar is acceptable for the 4 primary destinations.
- **Auth gate:** unauthenticated users land on Sign In; Ask AI may be previewable but booking requires auth.

---

## 6. Screens & flows

Each maps to existing backend endpoints — build against these, not new APIs.

### Home
Minimal hero + three entry cards: **Ask the Legal AI**, **Find Lawyers**, **Analyze a Document**. Below: short "how it works" strip and a trust note ("Educational information, grounded in Indian bare acts — not a substitute for a lawyer").

### Ask AI (hero screen) — `POST /api/chat/stream` (SSE), `/api/chat`
- Two-pane on desktop: message thread (left/center) + collapsible **context panel** (right) showing detected intent, retrieved statutes/case-law citations, and confidence.
- Streaming assistant messages (token-by-token). Markdown rendering; statute refs styled as mono chips.
- **Composer:** multiline input, send, attach (document), and quick-action chips: *Explain a law*, *Report a crime*, *Analyze document*, *Find a lawyer*.
- Intent-aware result blocks:
  - `find_lawyer` → inline lawyer cards with "View / Book".
  - `crime_report` → numbered guidance steps + relevant sections.
  - `document_analysis` → summary card (type, statutory checks, flagged defects).
- Session controls: new chat / clear (`DELETE /api/chat/session/{id}`), history.
- **Empty state:** greeting + capability list + example prompts.

### Document Analysis — `POST /api/chat/upload`, `/analyze-document`, `/validate-document`
Dropzone (PDF/DOCX/image, OCR handled server-side) or paste text. Result: classification badge, statutory-requirement checklist (pass/fail), defect list (warning-colored), extracted key fields. Show processing state clearly (OCR can be slow).

### Find Lawyers — `GET /api/lawyers`, `POST /api/lawyers/recommend`, `/api/chat/find-lawyer`
- **Directory:** searchable/filterable grid of lawyer cards (name, specialty, experience, rating, hourly rate, location, languages, availability). Filter by specialization (`/api/chat/specializations`) & location.
- **Recommend:** short guided form (case description / specialty / location) → ranked results.
- **Lawyer profile:** full bio, stats (cases, success rate, education), and a prominent **Book** CTA.

### Booking & Payment — `GET /api/bookings/client_token`, `POST /api/bookings/checkout`
Braintree Drop-in (sandbox). Summary of lawyer + amount, secure-payment reassurance, success → confirmation. Use `fake-valid-nonce` in test.

### My Bookings — `GET /api/bookings/user-bookings/{userId}`
List of confirmed consultations: lawyer, amount, status pill, transaction id. Clean empty state.

### Auth — `POST /api/auth/register`, `/login`, `GET /api/auth/me`
Minimal centered card. JWT stored client-side; restore session on load. Show `{message}`-shaped errors inline.

---

## 7. Component inventory

Buttons (primary/secondary/ghost/danger) · Input, Textarea, Select, Combobox · Card · Badge/Pill (specialty, status, confidence) · Avatar · Tabs · Dialog/Sheet/Drawer · Toast (sonner) · Dropzone · Chat bubble (user/assistant, streaming, loading) · Citation chip · Skeleton loaders · Empty state · Rating stars · Nav bar · Theme toggle.

Keep the Radix-primitive / shadcn-style component approach already in the repo; restyle to these tokens.

## 8. States & feedback

- **Loading:** skeletons for lists/cards; typing indicator + streaming for chat; explicit progress copy for uploads/OCR.
- **Empty:** friendly one-liner + primary action (never a blank panel).
- **Error:** inline near the cause; toast for transient failures; retry affordance. Surface backend `{message}` text.
- **Trust cues:** citation chips, confidence badge (`success`/`warning`), and a persistent "not legal advice" disclaimer near AI output.

## 9. Accessibility & responsive

- WCAG AA contrast in both themes; visible focus rings (`--accent`); full keyboard nav; ARIA on chat log (`aria-live=polite` for streamed tokens), dialogs, and forms.
- Breakpoints: `sm 640 · md 768 · lg 1024 · xl 1280`. Two-pane chat and lawyer grid collapse to single column below `lg`. Tap targets ≥44px.

## 10. Out of scope / notes for build

- Backend is fixed: FastAPI on `:8000`, base `VITE_API_URL` (default `http://localhost:8000/api`). Lawyer/booking JSON is **camelCase**; `client_token` returns **plain text**; auth/lawyer/booking errors are `{message}`.
- Ignore legacy leftovers in the current `App.tsx` (hardcoded `:5001`, MongoDB/Razorpay comments) — target the real endpoints above.
- No real payments/PII beyond Braintree sandbox; keep the "educational, not legal advice" framing visible.
