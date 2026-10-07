# Brag Plan: LawWeb

## What is this app?
LawWeb is an AI counsel for Indian law: ask in plain language and get an answer cited to the exact bare-act section, then book a vetted advocate, check a document against the statutes, and keep your cases, vault and hearings in one place.

## The angle
A quiet premium product film in LawWeb's own print-inspired look (warm parchment, deep ink, one antique-gold accent, Fraunces serif). The product's promise is *"Understand exactly where you stand."* The video proves that line by following one person's worry from start to finish: a bounced cheque. The question is cited to a real section (Section 138, NI Act), a Banking Law advocate is booked from the answer, and a rent agreement is checked against the Registration Act. The specificity of real Indian statutes is the brag. No generic AI imagery.

## Hook (first 2-3 seconds)
The hero line, verbatim, set big on parchment: "Understand exactly / where you *stand.*", with "stand." in gold italic. A thin gold rule draws in beneath it. Restraint is the hook: it should look like the opening of a well-printed book.

## Key moments (the middle)
- **The cited answer.** A real question is typed into the LawWeb composer. The "LawWeb · Counsel" answer streams in, and the citation (*Section 138, Negotiable Instruments Act, 1881*) settles in gold.
- **Answer → advocate.** "Lawyers who could help with this" appears under the answer with a real directory card (Vinaya Luthra · Banking Law · Delhi · ★ 4.9). The cursor clicks **Book**, a slot is picked, and a green "Confirmed" lands.
- **Read the fine print.** `Rent_Agreement.pdf` drops into the dropzone. The 3-layer pipeline ticks (Classified → Statutory checklist → Defects), then one flagged defect: lease over one year, so it must be registered (Registration Act, 1908 · s.17).
- **And everything after.** Vault, My Cases and the Legal Calendar arrive as three quiet cards: the matter stays organised after the first answer.

## Outro / punchline
Cut to the deep-ink closing panel from the site: the gold-outlined "L" mark, "LawWeb", and the site's own closing line, "Your first question is waiting to be answered." A small trust line beneath: "Grounded in Indian bare acts. Not a substitute for a lawyer." Music resolves; silence.

## User flow worth showing
1. **Ask:** type "A cheque I received has bounced. What can I do?" and a cited answer streams in.
2. **Act:** a lawyer is suggested inline, cursor → Book → slot → Confirmed.
3. **Verify:** upload an agreement → pipeline → a statutory defect flagged with its section.

## Tone
- Preset: polished
- Creative direction: quiet premium product film, like a well-printed legal book coming to life
- Interpretation: slow, confident reveals, soft crossfades (0.6-0.8s), generous parchment space, light-to-medium Fraunces, no bullets or hype words. Five scenes plus the outro (more than polished's usual 3-4, because the user asked for all four flows); scene 5 is a brief montage so the pace still breathes.

## Format: landscape — 1920x1080
## Duration: 25s

## Visual identity (from the project)
- Background: `#f4efe4` (parchment); surfaces `#fdfbf5`, `#efe8d9`
- Accent: `#8c6f2f` (antique gold); gold line `#c9ab63`; tint `#efe5cd`
- Text: `#211d15` ink / `#17130c` strong; muted `#6a6252`
- Ink panel (outro): bg `#211d15`, fg `#fbf7ec`
- Status: green `#4f7d52` on `#e6ecdd`; amber `#8f6a1a` on `#f6ecd6`
- Display font: Fraunces (Google Fonts, opsz 9..144, 400-700, italic)
- Reading serif: Newsreader; UI sans: Inter (all three self-hosted as woff2 in the composition)
- Contrast: read text uses `--text`/`--text-2`/`--muted`; gold only for large or bold text; avoid `--muted-2/-3` for anything that must be read
- Radii crisp (3-8px), hairline borders `#e5ddcd`, restrained shadows
- Strongest visual element: the hero headline with gold-italic "stand." and the gold rule; the chat answer card with "LawWeb · Counsel" label and the "L" brand mark

## Share copy (draft)
Introducing LawWeb: ask a legal question in plain language and get an answer cited to the exact Indian bare-act section, then book an advocate or check a document in the same place. Built with FastAPI, LangGraph, hybrid RAG over 90+ Indian acts, and React.

## Audio direction
- Role: warm, steady bed with sparse professional accents
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady and clean; the recommended polished track)
- Music treatment: start at 0 with a short fade-in (~0.8s), bed at ~0.30, fade out over the last ~1.5s so the outro ends near silence
- Music cue guidance: preset read (`assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.md`), ~110 BPM. Strong cue locks: **8.74s** (lawyer suggestion card lands in chat) and **22.37s** (outro closing line, strength 0.98). Beat grid: pipeline steps 15.29 / 15.84, finding 16.38; montage cards 19.10 / 19.66 / 20.19 (short title-only labels, then held together).
- Audio-reactive treatment: subtle; music RMS gently warms a soft gold glow behind the hero headline and the outro mark. No waveforms or bars.
- SFX posture: sparse, low-HF-risk, quiet (0.5-0.65)
- Audio-coupled moments: question typed (soft key ticks, thinned), cursor click on Book, "Confirmed" success accent, PDF drop, defect flag, one soft bell on the outro mark
- Restraint rule: no aggressive impacts, no glitch sounds, no SFX on every item; let the bed carry the transitions

## Storyboard

### Scene 1 — Hook — 0.0–2.8s (2.8s)
Parchment ground. Eyebrow "INDIAN LAW · RETRIEVAL-BACKED COUNSEL" (gold, tracked caps) fades up. Headline "Understand exactly / where you *stand.*" rises in by line, Fraunces large; "stand." gold italic. Gold rule draws left to right underneath. Hold ≥1.4s settled.
Sequential/interaction: two headline lines arrive one after the other
Audio intent: calm opening, music fades in
Audio-coupled idea: none (let the music open), optional very soft accent as the rule draws
Music: bed fades in
Transition mood: soft crossfade / gentle push in → Scene 2

### Scene 2 — Ask — 2.8–10.3s (7.5s)
Recreated LawWeb chat screen at video scale (UI ≈1.7× app size; nothing that must be read under ~24px). The composer types "A cheque I received has bounced. What can I do?" (≈3.3–4.6s) and it posts as a warm user bubble. An answer block with the "L" mark and "LAWWEB · COUNSEL" label streams in (Newsreader), finishing by ≈6.3s:
"An offence under **Section 138, Negotiable Instruments Act**. Send a demand notice within 30 days."
The citation phrase settles in gold/bold. Answer settled ≈6.3s → crossfade ≈10.0s (≈3.7s). At **8.74s** (strong cue) "Lawyers who could help with this" plus one lawyer card slides up: avatar "VL", **Vinaya Luthra**, "Banking Law · Delhi", ★ 4.9, [Profile] [Book].
Sequential/interaction: yes, question types by character, answer streams by word-group, then the lawyer card
Audio intent: focused, attentive; the product doing real work
Audio-coupled idea: thinned key ticks while typing; soft drop as the lawyer card lands (beat-locked 8.74s)
Transition mood: clean; camera eases toward the lawyer card → Scene 3

### Scene 3 — Book — 10.3–14.0s (3.7s)
Continues from the card: cursor moves to **Book**, click. A compact booking sheet "Book a consultation · Vinaya Luthra" with three slot chips ("Thu 10:00", "Thu 11:30", "Fri 16:00"). The cursor picks Thu 11:30, then **Confirm**, and a green "Confirmed · Thu, 11:30 AM" pill lands by ≈12.6s. Settled until crossfade ≈13.7s (≈1.1s).
Sequential/interaction: yes, simulated cursor: Book → slot → Confirm
Audio intent: decisive, satisfying
Audio-coupled idea: mouse clicks matched to each click; gentle success accent on Confirmed
Transition mood: soft crossfade → Scene 4

### Scene 4 — Read the fine print — 14.0–18.8s (4.8s)
Page title "Analyze a document" (Fraunces), with no sub-line. A file chip `Rent_Agreement.pdf` drops into the dashed dropzone (≈14.7s). Three pipeline steps tick on every-other beat, then hold: "Classified · Lease agreement" ✓ (≈15.29), "Statutory checklist" ✓ (≈15.84), then an amber finding card at ≈16.38s:
"Lease over one year — must be registered" with the citation on its own line: *Registration Act, 1908 · s.17*
Finding settled ≈16.4 → crossfade ≈18.5s (≈2.1s).
Sequential/interaction: yes, file drop, two steps tick, then the finding
Audio intent: careful, precise
Audio-coupled idea: soft drop on file land; quiet tick on the first step only; a muted accent on the defect flag
Transition mood: soft slide → Scene 5

### Scene 5 — Everything after — 18.8–21.5s (2.7s)
Three cards slide in left to right, **titles only** (Fraunces), with a small icon each: **Document Vault**, **My Cases**, **Legal Calendar**. Beats ≈19.10 / 19.66 / 20.19; full set settled ≈20.3 → crossfade ≈21.2 (≈0.9s, and the first card holds ≈2s).
Sequential/interaction: yes, 3 cards one by one, then hold
Audio intent: lift toward the close
Audio-coupled idea: one soft card-place on the first card, nothing else
Transition mood: dissolve into ink → Scene 6

### Scene 6 — Outro — 21.5–25.0s (3.5s)
Deep ink panel `#211d15`. Gold-outlined "L" mark, "LawWeb" wordmark (Fraunces, cream) and the small trust line "Grounded in Indian bare acts · Not a substitute for a lawyer" arrive together (≈21.8s). At **22.37s** (cue, strength 0.98) the closing line: "Your first question is waiting to be answered." Settled to 25.0 (≈2.6s). Nothing extends past 25.0.
Sequential/interaction: mark + wordmark, then the line
Audio intent: resolve, quiet confidence
Audio-coupled idea: one soft bell when the mark lands; music fades out over the last ~1.5s
Transition mood: end

**Music mood for this video:** steady, warm, understated (polished)
**Audio summary:** a clean bed fades in under the hero, carries the chat → booking → document flow with a few soft interaction sounds locked to real clicks and drops, and resolves under the ink outro with one quiet bell.
