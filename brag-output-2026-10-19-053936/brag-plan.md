# Brag Plan: LawWeb (technical cut, v3)

v3 (2026-10-19) answers review notes on v2; v1 (`brag-output-2026-10-14-151742/`) and v2 (`brag-output-2026-10-14-170958/`) are unchanged.
1. The 1.0 s first-token figure is the LLM call alone (prompt replay). It is labelled "LLM first token" and always paired with the live search + grading time measured in the capture.
2. The live clips carry one constant badge, "waits compressed N×", next to a real elapsed timer. There is no "real time" state.
3. PRs #64 and #65 merged on 2026-10-14, so no PR numbers appear on screen.
4. Answer lists render as nested lists with (i)/(ii) markers (PR #66). Both live clips were re-captured with it, on a backend with the rebuilt judgment, precedent and classifier indices.

v2 answered the user's notes on v1:
1. every metric re-verified and labelled with its test set, nothing misleading;
2. no name credit and no "academic project" anywhere (outro shows the repo link and email);
3. multilingual support shown (architecture + a live Hindi answer);
4. a real screen capture of an answer streaming on this laptop, with live nvidia-smi readings;
5. more architecture, balanced with metrics; ~90 s.

## What is this app?
LawWeb answers Indian legal questions, citing the exact bare-act section, with a LangGraph agent, hybrid retrieval and small trained classifiers around a 4B model that runs entirely on a 4 GB laptop GPU.

## The angle
Constraint-driven engineering, shown working and measured. The live clip proves it runs; the architecture scenes show how it fits; every number carries its test set.

## Hook
VRAM meter fills to the live nvidia-smi reading (3,465 / 4,096 MiB) under "Indian legal AI, on a 4 GB laptop GPU."

## Tone
- Preset: polished. Creative direction: a research paper typeset in LawWeb's print look, coming to life; calm, exact; narrated.

## Format: landscape 1920x1080 · Duration: ~90 s (user choice; overrides /brag's 15-25 s)

## Verified numbers (frozen 2026-10-14; the video uses only these)
| Claim | Value | Source / test set | Re-run today? |
|---|---|---|---|
| Live answer (English) | first text 14.7 s, done 25.6 s | capture `en5`, question "Can the police arrest me without a warrant?", fresh backend, warm | live |
| Live answer (Hindi) | done 32.3 s | capture `hi3`, same question in Hindi, with PR #65 | live |
| GPU during the answer | 3,465 / 4,096 MiB used; Ollama 3,380 MiB; util up to 100 %; 67-78 °C | nvidia-smi every 250 ms during `en5` | live |
| Statute index | 15,100 sections · 100 acts, rules & mapping tables | `faiss_index/unified/sections.json` | yes |
| Judgment index | 1,074,755 passages · 38,338 SC judgments · IVF4096-SQ8 + SQLite FTS5 | `faiss_index/judgments/meta.json` | yes |
| Retrieval | hit@1 0.744 · hit@3 0.897 · MRR 0.816 | `eval_retrieval_only.py --extended`, 39 hand-written questions with gold sections, production `retrieve_statutes` path | yes |
| IL-TUR statute identification | 38.35 macro-F1 (37.43 without near-dup cases) vs LeSICiN 28.08 · InLegalBERT 26.23 · GPT-4 23.99 (published) | official scorer, 13,019 test cases; data README | no (documented, leaderboard-scorer verified) |
| Intent routing | 0.915 routed accuracy (200 test messages); real forum questions 0.808; centroid baseline 0.780 at training time | `eval_routing.py --split test` | yes |
| Document type | 0.938 vs 0.815 regex (130 test docs: 110 hand-written, 10 real SC judgments, 10 real FIR scans); FIR scans 10/10 vs 8/10 | `eval_doc_type.py score --split test` | yes |
| Crime sections | hit@5 0.826 (ILSIC test, 350 real lay questions); dev 0.357 → 0.838 when added | PR #43 (scored once) | partial (hit@2 0.72 at 25/350, stopped) |
| Document review latency (qwen3:4b) | rent agreement 247-269 → 99 s; FIR 220-224 → 103 s; 3 LLM calls → 2; 0 fallbacks in 4 runs (was 4 of 6) | PR #55 | no |
| Streaming | first status label 0.8-2.2 s | PR #57 | no |
| Model selection | 7 models fit 4 GB; qwen3:4b-instruct quality 0.872 vs 0.876 (95 % interval −0.030…+0.024); answer call median 7.0 vs 21.4 s; first token 1.0 vs 12.6 s | `results/model_bench/scores.json`, prompt replay, 37 scored items × 2 seeds, 798 calls | no (committed results) |
| Languages | 16 Indian languages detected end to end (of 23 mapped) | fastText lid.176 labels ∩ IndicTrans2 map | yes |
| Tests | 370 unit tests passing on main (383 with PR #65) | `pytest tests/unit` | yes |

Not shown: the v1 cheque mock-up (the real app answers that question wrongly; reported), absolute faithfulness/recall, crime-report routing recall.

## Voiceover script (Kokoro af_heart, 1.1x, one clip per sentence)
1. LawWeb answers Indian legal questions, citing the exact section, on a four gigabyte laptop GPU.
2. Here it is, live: about fifteen seconds of searching, then the answer streams in and its citations are checked.
3. Everything runs locally. The model has the GPU; embeddings, reranking, classifiers and translation share the CPU.
4. A LangGraph agent routes each message. Legal questions loop: retrieve, grade, widen the search if evidence is weak, then generate and verify.
5. Search is hybrid: keyword and dense, fused and reranked. On our test questions, the right section is in the top three nine times in ten.
6. Hindi and fifteen other Indian languages are translated in and out with IndicTrans2, with citations masked so they survive.
7. On the IL-TUR benchmark, our statute classifier beats the published baselines by ten points.
8. Small classifiers route messages, type documents and suggest crime sections.
9. Document review plans once per document type and streams its analysis, more than twice as fast as before.
10. Seven small models were benchmarked with the rule fixed in advance. Timed on the LLM call alone: first token in one second, not twelve, at equal quality.
11. Every byte is budgeted, so the model never leaves the GPU.
12. Every number here was measured. The code is on GitHub.

## Storyboard (≈ 50 % architecture, 40 % metrics, 10 % live product)
1. **Hook** — VRAM meter (live reading) + headline.
2. **Live** — real capture in a browser card: typing at 1×, the search wait compressed (constant "waits compressed N×" badge), the answer streaming at 1×, beside a real elapsed timer and the nvidia-smi panel. A summary line gives the measured split: search + grading, then the model's first token.
3. **System architecture** — client → FastAPI (SSE) → LangGraph → GPU lane (Ollama, qwen3 4B) and CPU lane (BGE-M3, reranker, InLegalBERT statute classifier, routing/doc/crime heads, fastText + IndicTrans2) → data (unified statute index, judgment index, case law, Postgres chat memory).
4. **Agent graph** — real node names and the corrective loop.
5. **Retrieval** — BM25 + BGE-M3 → RRF → rerank; index sizes; hit@3 0.90 (39 questions, re-run).
6. **Multilingual** — detect → mask citations → IndicTrans2 → English graph → back-translate → restore; live Hindi clip with the same badge and timer.
7. **IL-TUR** — bars vs published baselines.
8. **Classifiers** — routing / document type / crime sections with honest test-set labels.
9. **Document pipeline** — THINK cached per document kind, overlapped with retrieval; ANALYZE streams; per-document latency (qwen3:4b).
10. **Model selection** — table, pre-committed rule; first token and median time labelled "LLM call", with a note that search + grading run before it (~N s live) and a source line saying retrieval is excluded.
11. **Hardware budget** — live VRAM reading + four fit decisions.
12. **Outro** — results board, github.com/anubhab7111/lawWeb, anubhabdas7111@gmail.com.

## Audio
Narration leads; vol-12 bed carved against the voiceover group; sparse SFX (send click, bar landing, outro bell).
