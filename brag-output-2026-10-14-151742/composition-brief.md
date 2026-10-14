# Hyperframes Composition Brief: LawWeb (technical cut)

## Objective
Create a narrated ~75 s technical walkthrough of LawWeb for YC founders: an academic project pitched for a job. Lead with the value (cited Indian-law answers from a model on a 4 GB laptop GPU), then show the implementation and optimizations, and put the best measured test results on screen.

## Output
- Composition directory: `brag-output-2026-10-14-151742/composition/`
- Rendered video: `brag-output-2026-10-14-151742/brag.mp4`
- Format: landscape — 1920x1080, 30 fps
- Duration: 75.9 s (user chose 60-75 s, overriding /brag's 15-25 s default; scene lengths follow the narration)

## Source Material
- Project root: the lawWeb repo (this worktree, rebased on main at c75c094)
- Primary files read: `server/app/chatbot.py` (LangGraph graph, `gq_grade` / `gq_rewrite` / `gq_verify`), `server/app/config.py` (BGE-M3, bge-reranker), `server/app/data/README.md` (judgment index, IL-TUR results), `server/app/data/faiss_index/unified/sections.json` (15,100 sections, 100 sources), `metrics_20261004_153111.json` (retrieval sweep), PR #43 / #55 / #57 / #63 bodies, commits 1c48047 / d9181d2 / 2572866, the earlier brag film's chat UI
- Product name: LawWeb
- Strongest claim: statute classifier at 38.35 macro-F1 on IL-TUR vs the published LeSICiN baseline (28.08)
- Key UI to recreate: the Ask AI chat answer card citing Section 138, NI Act; the document page with its real streaming status labels
- Copy that must appear verbatim:
  - "Checking the law that applies…", "Reading the document…", "Writing the analysis…" (real status labels)
  - Real LangGraph node names: `classify_intent`, `gq_plan`, `gq_retrieve`, `gq_grade`, `gq_rewrite`, `gq_generate`, `gq_verify`
  - Contact: github.com/anubhab7111/lawWeb · anubhabdas7111@gmail.com · "An academic project by Anubhab Das"

## Creative Direction
- Tone preset: polished
- Creative direction: a research paper typeset in LawWeb's print look, coming to life; calm, exact; narrated
- Interpretation: one figure per scene ("Fig. N · topic"), one hero number per figure held long enough to read, small source lines with test-set sizes, fade-through-paper transitions
- Angle: constraint-driven engineering, measured — see `brag-plan.md`
- Hook: VRAM meter filling to 3.65 / 4.00 GB under "Indian legal AI, on a 4 GB laptop GPU."
- Outro: results board of the six best numbers, then LawWeb mark, name, GitHub and email on the ink panel
- Avoid: generic SaaS language, abstract filler, "SOTA" claims, before/after pairs from different test sets, absolute faithfulness/recall scores

## Visual Identity
- Background `#f4efe4`, surfaces `#fdfbf5` / `#efe8d9`, text `#211d15` / `#17130c`, accent `#8c6f2f`, gold line `#c9ab63`, ink panel `#211d15` / `#fbf7ec`
- Fonts: Fraunces (display, tabular figures), Newsreader (reading serif), Inter (UI), IBM Plex Mono (code identifiers) — all self-hosted woff2

## Storyboard
1. Hook — 6.0 s — VRAM meter + headline
2. Product — 7.8 s — cheque question → answer citing s.138 NI Act → real status label "Checking citations against the statutes…" (then clears) + `gq_verify` note
3. Agent graph — 10.6 s — intent fan-out and the corrective loop, stack strip
4. Hybrid retrieval — 9.3 s — BM25 + BGE-M3 → RRF → rerank; 15,100 sections; 1,074,755 passages; hit@3 0.94
5. IL-TUR — 7.8 s — bars 23.99 / 26.23 / 28.08 / 38.35, +10.27 bracket
6. Trained heads — 7.4 s — 0.780→0.930, 0.815→0.938, 0.357→0.838
7. Document latency — 9.1 s — status labels, 3 calls vs 2 calls on one axis, ~250 s → 99 s
8. Model selection — 9.1 s — 7-model table with gates, pre-committed decision rule, 12.6 s → 1.0 s first word
9. Hardware budget — 6.6 s — 3.65 / 4.00 GB meter and four fit decisions
10. Results + outro — 7.6 s — six metric tiles, 370 unit tests, identity and contact

## Audio
- Audio role: narration leads; steady ducked bed; three sparse accents
- Voice: Kokoro `af_heart` at 1.1x via `hyperframes tts`, one clip per sentence (`assets/vo/*.wav`), grouped as `voiceover`; name not spoken (Kokoro mispronounces it)
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`, volume lane 0 → 0.3 over 1.2 s, held, → 0 over the last 1.5 s; voiceover carve (strength 0.8) against the `voiceover` group
- Music cue guidance: narration sets the pace; no beat locks used
- Audio-reactive treatment: subtle; music RMS (`assets/audio-data.js`, 30 fps) drives the gold glow behind the hook and the outro mark
- SFX: `click_002` on the chat send (7.28 s), `drop_002` as the LawWeb bar starts (33.5 s), `bong_001` as the outro mark lands (71.0 s); none under a spoken number

## Hyperframes Instructions
Modular composition: thin `index.html` (slots, narration, music, SFX) + one sub-composition per scene in `compositions/`. Shared tokens and `@font-face` in `theme.css`. `npx hyperframes check` must pass with zero findings before render.
