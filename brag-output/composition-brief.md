# Hyperframes Composition Brief: LawWeb

## Objective
Create a short, polished launch-style brag video for LawWeb, an AI counsel for Indian law.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080, 30fps
- Duration: 25.0 seconds (root `data-duration="25"`; no clip or fade past 25.0)

## Source Material
- Project root: repo root (`client/` React app, `server/` FastAPI)
- Primary files read: `client/src/styles/theme.css`, `client/src/components/Home.tsx`, `AskAI.tsx`, `LawyerCard.tsx`, `DocumentAnalysis.tsx`, `Vault.tsx`, `MyCases.tsx`, `LegalCalendar.tsx`, `README.md`, `server/app/data/lawyers.json`
- Product name: LawWeb
- Tagline / strongest claim: "Understand exactly where you stand."
- Key UI to recreate: the chat answer block ("L" mark + "LAWWEB · COUNSEL" label + serif answer) with an inline lawyer card; the document dropzone + pipeline; the ink closing panel
- Copy that must appear verbatim:
  - "Understand exactly / where you stand." (stand. in gold italic)
  - "Indian Law · Retrieval-backed Counsel" (eyebrow)
  - "Lawyers who could help with this"
  - "Analyze a document"
  - "Your first question is waiting to be answered."

## Creative Direction
- Tone preset: polished
- Creative direction: quiet premium product film, like a well-printed legal book coming to life
- Interpretation: slow, confident reveals; soft crossfades 0.6-0.8s; generous parchment space; no hype words, no bullets
- Angle: one person's worry (a bounced cheque) followed through the product: cited answer → book an advocate → check a document → stay organised → close
- Hook: the hero headline set big on parchment with a gold rule drawing in
- Outro / punchline: ink panel, "L" mark, "LawWeb", "Your first question is waiting to be answered."
- Avoid:
  - Generic SaaS language, AI sparkle/robot imagery, abstract filler
  - Redesigning the brand: stay in parchment / ink / antique gold
  - Rounded bubbly SaaS cards with heavy shadows (the brand uses hairlines and crisp 3-8px radii)

## Visual Identity
- Background: `#f4efe4`; surfaces `#fdfbf5` / `#efe8d9`; hairline `#e5ddcd` / `#dcd2bf`
- Text: `#211d15` (ink), `#17130c` (strong), `#3b352a` (text-2), `#6a6252` (muted, the lightest allowed for read text)
- Accent: `#8c6f2f` gold (large/bold only), `#c9ab63` gold line, `#efe5cd` gold tint
- User bubble: bg `#ece4d1`, fg `#3a3122`
- Status: green `#3c5e3f` on `#e6ecdd`; amber `#8f6a1a` on `#f6ecd6`
- Ink panel: `#211d15` bg, `#fbf7ec` fg
- Display font: Fraunces (variable, opsz + italic); reading serif: Newsreader; UI: Inter. Self-host all as woff2 in `assets/fonts/`.
- Scale: recreated UI at ~1.6-2× the app's px sizes; nothing that must be read below ~24px.
- If any `theme.css` rules are reused, strip CSS `animation:` rules (they are wall-clock, not timeline-driven).

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Hook — 0.0–2.8s — eyebrow + two-line headline + gold rule
2. Ask — 2.8–10.3s — typed question, streamed cited answer (Section 138, Negotiable Instruments Act), lawyer card at 8.74s
3. Book — 10.3–14.0s — cursor: Book → slot Thu 11:30 → Confirm → green "Confirmed"
4. Read the fine print — 14.0–18.8s — "Analyze a document", PDF drop, 2 pipeline ticks, amber finding "Lease over one year — must be registered / Registration Act, 1908 · s.17" at ~16.38s
5. Everything after — 18.8–21.5s — three title-only cards: Document Vault, My Cases, Legal Calendar
6. Outro — 21.5–25.0s — ink panel, mark + wordmark + trust line, closing line at 22.37s

## Audio
- Audio role: warm, steady bed with sparse professional accents
- Audio arc: fade in under the hook, steady through the flow, soft bell on the outro mark, fade to near silence by 25.0
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` at ~0.30
- Music treatment: ~0.8s fade-in, ~1.5s fade-out ending at 25.0
- Music cue guidance: preset `<skill-dir>/assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json` (~110 BPM). Locks: 8.74s (lawyer card), 22.37s (closing line). Beat grid: 15.29 / 15.84 / 16.38 (pipeline + finding), 19.10 / 19.66 / 20.19 (cards).
- Audio-reactive treatment: subtle; RMS warms a soft gold glow behind the hero headline and the outro mark. No visualizer graphics.
- Audio-coupled moments:
  - Scene 2 typing — thinned soft key ticks
  - Scene 2 lawyer card — soft drop at 8.74s
  - Scene 3 — mouse clicks on Book / slot / Confirm; gentle success on Confirmed
  - Scene 4 — soft drop on file, muted accent on the finding
  - Scene 5 — one soft card-place on the first card
  - Scene 6 — one soft bell on the mark
- SFX selection guidance: polished = minimal; low-HF-risk files from `sfx-analysis.md`; volumes 0.5-0.65
- Exact SFX choice: decided after the animation exists
- Audio files: copied into `brag-output/composition/assets/`

## Hyperframes Instructions
Load `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`, `hyperframes-keyframes` and `hyperframes-cli`. /brag is its own workflow: do not enter the `hyperframes` entry-point intent interview and do not route into its generic promo / launch-video workflow.

Requirements:
- Show real UI / copy from LawWeb (chat answer, lawyer card, document pipeline).
- Keep all text readable; respect the reading-time holds in the plan.
- 25.0s total.
- Include music + SFX.
- Run `npx hyperframes check` before render.
