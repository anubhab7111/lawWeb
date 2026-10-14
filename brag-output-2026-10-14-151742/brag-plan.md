# Brag Plan: LawWeb (technical cut)

## What is this app?
LawWeb answers Indian legal questions with the exact bare-act section cited, using a LangGraph agent, hybrid retrieval and trained classifiers around a small local LLM that runs entirely on a 4 GB laptop GPU.

## Audience and purpose
YC founders watching an academic project as a hiring pitch. They should come away thinking: this person measures before tuning, fits real ML into hard constraints, and ships working software. Value first (cited answers on a laptop), then the engineering that makes it possible, then the measured results.

## The angle
**Constraint-driven engineering, measured.** Every optimization in LawWeb exists because the whole stack (embeddings, reranker, 1M-passage index, LLM) has to fit a 4 GB RTX 3050 and 15 GB of RAM. The video reads like a well-typeset research paper coming to life in LawWeb's own print look: each beat is a "figure" with a caption, one hero number, and a small source line (PR number, test-set size). The specificity is the brag: real node names from the graph, real model names, real benchmark baselines, real before/after numbers from the repo's PRs.

## Hook (first 2-3 seconds)
A VRAM meter fills to **3.65 / 4.00 GB**, labelled "qwen3 4B · 100% on GPU". Big Fraunces line lands beside it: "Indian legal AI, on a *4 GB* laptop GPU." The narration opens on the same claim. Constraint first: it explains why every later optimization exists.

## Key moments (the middle)
- **The product works.** A plain-language question gets a streamed answer citing *Section 138, Negotiable Instruments Act, 1881*, with a small "citation verified" mark.
- **The agent graph.** The real LangGraph topology draws itself: `classify_intent` fans to six paths; the general path runs `plan → retrieve → grade → rewrite ↺ → generate → verify`.
- **Hybrid retrieval.** BM25 + BGE-M3 dense → RRF fusion → cross-encoder rerank, over 15,100 statute sections and 1,074,755 Supreme Court passages. Result: **hit@3 0.94**.
- **Beats the published benchmark.** IL-TUR statute identification bars: GPT-4 23.99, InLegalBERT 26.23, LeSICiN 28.08, **LawWeb 38.35**.
- **Small trained heads.** Intent routing 0.780 → **0.930**; document type 0.815 → **0.938**; crime sections hit@5 0.357 → **0.838**.
- **Latency.** Document review ~250 s → **99 s**; model benchmark: first visible word 12.6 s → **1.0 s** at equal quality.
- **Every byte budgeted.** `num_ctx 8192` = largest window that keeps the model fully in VRAM; embeddings and reranker on CPU; reranker fine-tune at 2.35 GB peak; resumable, thermal-guarded GPU jobs.

## Outro / punchline
A results board (the best measured numbers, held), then the deep-ink closing panel: "L" mark, **LawWeb**, "An academic project by Anubhab Das", `github.com/anubhab7111/lawWeb`, `anubhabdas7111@gmail.com`, and a stack line: FastAPI · LangGraph · FAISS · BGE-M3 · Ollama · React.

## User flow worth showing
1. **Ask:** type "A cheque I received has bounced. What can I do?", and the answer streams in citing Section 138, NI Act, with a verified mark.
2. **Check a document:** the "Analyze a document" screen streams its real status labels ("Reading the document…", "Checking the law that applies…", "Writing the analysis…") while the latency figure lands.

(Product ≈ 25% of runtime; engineering ≈ 75%, as the user chose.)

## Tone
- Preset: polished
- Creative direction: a research paper typeset in LawWeb's print look, coming to life; calm, exact, confident; narrated
- Interpretation: one figure per scene, one hero number per figure, held long enough to read; soft crossfades/slides; figure captions ("Fig. 3 — Hybrid retrieval") and source lines give the academic register; no hype words; the numbers do the bragging. More scenes than polished's usual 3-4 because the user asked for a 60-75 s technical walkthrough.

## Format: landscape — 1920x1080
## Duration: 75.9 s (user chose 60-75 s; overrides /brag's 15-25 s default). Scene lengths follow the generated narration.

## Visual identity (from the project)
- Background: `#f4efe4` (parchment); surfaces `#fdfbf5`, `#efe8d9`
- Accent: `#8c6f2f` (antique gold), dark `#745a22`; line `#c9ab63`; tint `#efe5cd`
- Text: `#211d15` ink / `#17130c` strong; secondary `#3b352a`; muted `#6a6252`
- Ink panel: bg `#211d15`, fg `#fbf7ec`
- Status: green `#3c5e3f` on `#e6ecdd`; amber `#7a5a14` on `#f6ecd6`
- Display font: Fraunces (numbers in tabular lining figures); reading serif: Newsreader; UI sans: Inter; add a monospace (IBM Plex Mono or JetBrains Mono) only for code identifiers (`gq_verify`, `num_ctx`)
- Strongest visual elements: the gold-italic emphasis word in Fraunces headlines, the "LawWeb · Counsel" answer card with the "L" mark, the ink closing panel

## Verified numbers (all checked against the repo on 2026-10-14; show the n / source)
| Claim | Value | Source |
|---|---|---|
| Statute index | 15,100 sections · 100 acts, rules & mapping tables | `faiss_index/unified/sections.json` |
| Judgment index | 1,074,755 chunks from 38,338 SC judgments · IVF4096-SQ8 FAISS + SQLite FTS5 | `server/app/data/README.md` |
| Retrieval sweep (34 queries, 2026-10-04) | hit@1 0.79 · hit@3 0.94 · MRR 0.86 | `metrics_20261004_153111.json` |
| IL-TUR LSI, 13,019 test cases, official scorer | 38.35 macro-F1 (37.43 without near-dups) vs LeSICiN 28.08, InLegalBERT 26.23, GPT-4 0-shot 23.99 | data README, PR #40 |
| Intent routing (200-message test) | 0.930 vs 0.780 nearest-centroid; non-legal 1.000 vs 0.525 | commit 1c48047 |
| Document type (130 docs) | 0.938 vs 0.815 regex; real FIR scans 10/10 | commit d9181d2 |
| Crime sections (ILSIC real lay questions) | dev hit@5 0.357 → 0.838; test 0.826 | PR #43 |
| Document review latency (same docs, model fully on GPU) | rent agreement 247–269 s → 99 s; FIR 220 s → 103 s; 3 LLM calls → 2; 0 fallbacks (was 4/6) | PR #55 |
| Streaming | status label in 0.8–2.2 s, header ~20 s | PR #57 |
| Model benchmark (replay, 43 items × 2 seeds, 7 models) | qwen3:4b-instruct: first word 12.6 s → 1.0 s, median answer 21.4 s → 7.0 s, quality 0.876 vs 0.872 (tie) | PR #63 |
| End-to-end 27 cases | intents 27/27; find-lawyer median 21.1 s → 5.5 s | PR #63 |
| VRAM fit | num_ctx 8192 keeps qwen3 4B 100% on GPU (3.65 GB, 0 CPU spill) | PR #27 notes |
| Reranker fine-tune | frozen word embeddings → 2.35 GB peak on 4 GB | commit 2572866 |
| Tests | 370 unit tests | `pytest --collect-only` |

Do not show: absolute faithfulness/recall scores, routing crime recall, or any before/after pair measured on different query sets.

## Share copy (draft)
LawWeb, my academic project: Indian legal answers cited to the exact section, from a 4B model on a 4 GB laptop GPU. Hybrid RAG over 1M+ Supreme Court passages, a statute classifier at 38.35 macro-F1 on IL-TUR (published best 28.08), and document review cut from ~250 s to 99 s.

## Audio direction
- Role: narration leads; warm, steady bed underneath; very sparse accents
- Voice: Kokoro `af_heart` (default, highest-quality Kokoro voice), generated per scene so scene lengths follow the voice
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady and clean)
- Music treatment: short fade-in under the hook, ducked to ~0.13 while narration plays, lifts to ~0.28 after the last spoken line over the outro, fades out over the final ~1.5 s
- Music cue guidance: preset `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.md` (~110 BPM). Narration sets the pace, so cue locks are optional; use at most one strong cue for the outro mark if it falls within ±0.15 s of the planned time.
- Audio-reactive treatment: subtle; music RMS gently warms the gold glow behind the hook and the outro mark only
- SFX posture: sparse, quiet (0.45-0.6), low HF risk
- Audio-coupled moments: thinned key ticks while the question types; one soft drop when the IL-TUR LawWeb bar lands; one soft bell on the outro mark
- Restraint rule: never put an SFX under a spoken number; no SFX on every card or node

## Voiceover script (as generated: Kokoro af_heart, speed 1.1, one clip per sentence)
1. LawWeb answers Indian legal questions on a four gigabyte laptop GPU.
2. Ask in plain language. / Answers cite the exact section, and every citation is checked before you see it.
3. A LangGraph agent routes each message. / Legal questions run a corrective loop: retrieve, grade, rewrite if the evidence is weak, then generate and verify.
4. Retrieval is hybrid: keyword and dense search, fused, then reranked. / The right section is in the top three, ninety-four percent of the time.
5. Our statute classifier beats the published IL-TUR baselines by more than ten points.
6. Slow, brittle prompt steps became small trained classifiers.
7. Document review now makes two model calls instead of three, and streams as it writes. / About two hundred fifty seconds became ninety-nine.
8. We benchmarked seven small models, with the decision rule fixed before any of them ran. / The winner shows its first word in one second, not twelve.
9. And every byte is budgeted, so the model never leaves the GPU.
10. All of it, measured on held-out tests. / LawWeb. An academic project, open on GitHub.

(171 words, 59.7 s of speech. Spoken numbers are rounded; the screen shows exact figures. Kokoro phonemizes "IL-TUR" as "ill-tur". The name is shown, not spoken: Kokoro reads "Anubhab" as "AN-uh-b-hab". Scenes 10 and 11 were merged into one results + identity card to stay near 75 s.)

## Storyboard
Durations are targets; the final cut follows the generated narration (+0.8-1.5 s visual tail per scene).

### Scene 1 — Hook: the constraint — ~5.5 s
Parchment. Left: a slim VRAM meter (rounded bar) fills to 3.65 of 4.00 GB; label "RTX 3050 · 4 GB VRAM" above, "qwen3 4B · 100% on GPU" below in mono. Right: Fraunces headline in two lines, "Indian legal AI," / "on a *4 GB* laptop GPU." ("4 GB" gold italic). Gold rule draws under it.
Sequential/interaction: meter fills, then the two headline lines
Audio intent: calm, assured opening; music fades in under the voice
Audio-coupled idea: none (voice carries it)
Transition mood: soft crossfade → Scene 2

### Scene 2 — Product: a cited answer — ~7 s
Recreated LawWeb chat window (from the existing film: nav, composer, user bubble, "LawWeb · Counsel" answer with "L" mark). Question types: "A cheque I received has bounced. What can I do?" Answer streams: "An offence under **Section 138, Negotiable Instruments Act, 1881**. Send a demand notice within 30 days." Citation settles in gold, then a small green chip "✓ citation verified" appears beside it.
Caption (small, top-left): "Fig. 1 — Every answer cites the bare act"
Sequential/interaction: typing, streaming, chip
Audio-coupled idea: thinned key ticks while typing (stop before the voice's key words)
Transition mood: the window pulls back and dissolves into the graph → Scene 3

### Scene 3 — The agent graph — ~8.5 s
"Fig. 2 — LangGraph agent". Node diagram draws left to right with real node names in mono pills: `classify_intent` → six branch labels (document · crime report · find lawyer · general · non-legal · clarify); the general branch expands into the loop `gq_plan → gq_retrieve → gq_grade → gq_generate → gq_verify`, with a curved gold arrow `gq_grade → gq_rewrite ↺ gq_retrieve` and a second from `gq_verify` back to `gq_retrieve`. A thin stack strip at the bottom: React (Vite) · FastAPI + SSE streaming · LangGraph · Ollama (local).
Sequential/interaction: nodes appear in flow order; the loop arrows draw last, timed to "rewrite if the evidence is weak"
Transition mood: slide → Scene 4

### Scene 4 — Hybrid retrieval — ~9 s
"Fig. 3 — Hybrid retrieval". Pipeline row: BM25 + BGE-M3 dense → RRF fusion → cross-encoder rerank (bge-reranker) → top-k. Beneath, two index cards: "15,100 statute sections · 100 acts, rules & mapping tables" and "1,074,755 Supreme Court passages · IVF4096-SQ8 + FTS5". Hero number lands on the right: **0.94** hit@3, with "MRR 0.86 · 34-query sweep" beneath.
Sequential/interaction: pipeline stages light in order; index cards; then the hero number (held ≥2.5 s)
Transition mood: soft crossfade → Scene 5

### Scene 5 — IL-TUR benchmark — ~7.5 s
"Fig. 4 — IL-TUR statute identification (macro-F1)". Four horizontal bars grow from a shared axis: GPT-4 0-shot 23.99 · InLegalBERT 26.23 · LeSICiN 28.08 (muted ink) · **LawWeb 38.35** (gold, last, labelled "InLegalBERT classifier, chunked"). A bracket marks "+10.27 over best published". Fine print: "13,019 test cases · official leaderboard scorer · 37.43 with near-duplicate cases removed".
Sequential/interaction: baselines grow together, LawWeb bar grows last and longer
Audio-coupled idea: one soft drop as the LawWeb bar lands
Transition mood: soft crossfade → Scene 6

### Scene 6 — Small trained heads — ~7.5 s
"Fig. 5 — Trained heads on BGE-M3 embeddings". Three cards in a row, each with before → after and its test set:
- Intent routing: 0.780 → **0.930** (200-message test)
- Document type: 0.815 → **0.938** (130 documents, vs regex)
- Crime sections, hit@5: 0.357 → **0.838** (real lay questions, ILSIC)
Sequential/interaction: cards arrive left to right, then all three hold together ≥2.5 s
Transition mood: slide → Scene 7

### Scene 7 — Document review latency — ~8.5 s
"Fig. 6 — Document review". Left: the product's "Analyze a document" card with `Rent_Agreement.pdf`; its real streaming status labels appear one after another ("Reading the document…" → "Checking the law that applies…" → "Writing the analysis…"). Right: two timeline bars on the same scale: before = THINK · OBSERVE · ANALYZE (3 calls) spanning ~250 s; after = THINK (cached, runs during retrieval) · ANALYZE (2 calls) ending at 99 s. Hero: "~250 s → **99 s**", with "first status in ~1 s" beneath.
Sequential/interaction: status labels stream; bars draw; hero number lands
Transition mood: soft crossfade → Scene 8

### Scene 8 — Model benchmark — ~8.5 s
"Fig. 7 — Seven small models, one 4 GB GPU". A compact list of the 7 tested models (qwen3:4b, qwen3:4b-instruct, llama3.2:3b, granite4.2:3b, phi4-mini, qwen2.5:3b-instruct, nemotron-mini:4b); a stamp "decision rule committed before any run". The winner row highlights: qwen3:4b-instruct. Two hero stats: first visible word **12.6 s → 1.0 s**; median answer **21.4 s → 7.0 s**; and "quality 0.876 → 0.872 (tie)". Fine print: "prompt replay · 43 items × 2 seeds · 798 calls".
Sequential/interaction: list appears, winner highlights, stats land and hold
Transition mood: slide → Scene 9

### Scene 9 — Every byte budgeted — ~6 s
"Fig. 8 — Fitting in 4 GB". The VRAM meter from the hook returns, now segmented: qwen3 4B weights + KV cache at `num_ctx 8192` = 3.65 GB (fully on GPU). Four short chips arrive around it: "Embeddings + reranker → CPU", "One num_ctx → no 2 s model reloads", "Reranker fine-tune: 2.35 GB peak", "Resumable, thermal-guarded GPU jobs".
Sequential/interaction: meter, then chips 2-by-2 and hold
Transition mood: soft crossfade → Scene 10

### Scene 10 — Results board — ~4.5 s
"Measured on held-out tests". A 3×2 grid of the best numbers, held: 38.35 IL-TUR macro-F1 · 0.94 hit@3 · 0.930 intent routing · 0.938 document type · 99 s document review · 1.0 s first word. A thin line under it: "370 unit tests".
Transition mood: dissolve into ink → Scene 11

### Scene 11 — Outro — ~5 s
Ink panel `#211d15`. Gold-outlined "L" mark, "LawWeb" (Fraunces, cream), "An academic project by Anubhab Das", then `github.com/anubhab7111/lawWeb` and `anubhabdas7111@gmail.com` (Inter/mono, cream at readable size), stack line in muted cream. Hold ≥2.5 s after the last element lands. Music lifts after the last spoken word and fades out.
Audio-coupled idea: one soft bell when the mark lands

**Music mood for this video:** steady, warm, understated (polished)
**Audio summary:** a calm voice carries the argument over a ducked, steady bed; a few quiet accents mark real interactions and the benchmark payoff; the music lifts gently under the ink outro and fades to silence.
