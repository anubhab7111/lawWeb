# Data layout

Source and derived corpus data no longer live in this directory. They are on the
corpus drive, `LAWWEB_CORPUS_ROOT` (set in `server/.env`), and only the vector /
lexical **indices** stay in the repo, under `faiss_index/`.

What remains here: `lawyers.json` (seed fixture for `init_db`), `legal_ontology.json`
and `case_law_manifest.py` (runtime), `section_maps.json` (IPC->BNS and CrPC->BNSS
section correspondences, from the official tables), `vault/` (runtime upload storage,
gitignored), and `faiss_index/` (indices).

Serving never needs the drive: `unified` / `case_law` / `precedent` indices are
self-contained (chunk text and metadata are inside them). If the drive is unmounted
the staleness check is skipped and the existing index is loaded; only rebuilds and
ingestion need it.

## Corpus drive

`python -m app.ingest.corpus_layout --verify` (from `server/`) checks it. Layout:

```
statutes/           bare_acts/<domain>/*.pdf, mappings/, rules/, guides/, notifications/, explanatory/
case_law/curated/   landmark judgments (JSON)
judgments/sc/       pdf/ text/ clean/ chunks/ metadata/ manifest/   (Supreme Court, 1950-2026)
iltur/lsi/          IL-TUR export (CC BY-NC-SA 4.0 - non-commercial)
indiankanoon/       API response cache
builds/             embedding shards, eval + tuning caches, QA and decontamination reports
manifest/           file hashes, download ledgers
quarantine/         files rejected by a quality gate, each with a reason
```

Layers flow one way: raw PDF -> `text/` -> `clean/` -> `chunks/`. The chunked Supreme
Court corpus (38,338 judgments, 1,074,755 chunks) is indexed in full in
`faiss_index/judgments/` (IVF4096-SQ8 FAISS + SQLite FTS5); chat retrieves passages from it
and reranks them together with the curated landmark cases.

## Regenerating (from `server/`, conda env `legal_chatbot_env`)

| Step | Command |
|---|---|
| SC judgments (verified, resumable) | `python -m app.ingest.download_sc` |
| Extract + clean + chunk judgments | `python -m app.ingest.process_judgments` |
| IL-TUR export (needs HF token, gated) | `python -m app.ingest.iltur_export` |
| Precedent index (train+dev) | `EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_precedent_index` |
| Section maps (IPC->BNS, CrPC->BNSS) | `python -m app.ingest.section_maps` |
| Decontaminate judgments vs IL-TUR test/dev | `python -m app.ingest.decontaminate [--split dev]` |
| Judgment index (all chunks, resumable) | `EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_judgment_index` |
| Statute classifier (InLegalBERT) | `python train_lsi_classifier.py`, then `python -m app.ingest.export_statute_classifier` |
| Label reranker (fine-tuned cross-encoder) | `python train_label_reranker.py --cases 4000 --tag v1` |
| Statute + case-law indices | `python rebuild_rag_indices.py --all` |
| IL-TUR retrieval eval | `python eval_iltur_retrieval.py --split test --tag <name>` |
| Cache retrieval output for offline tuning | `python tune_iltur_retrieval.py cache ...` |
| Official IL-TUR scores (per system, sharded) | `python score_iltur.py <system> --split dev\|test` |
| Report + dev-fitted fusion | `python report_iltur.py classifier precedent_trainmem --fuse` |
| Check with the leaderboard's own scorer | `python verify_iltur_leaderboard.py` |

Run GPU builds with Ollama idle (see hardware constraints in `CLAUDE.md`).

## Evaluation validity

IL-TUR `lsi` labels are **bare section numbers**. The dataset attaches IPC text to each
one, but the cases beneath mix IPC and CrPC (label 482 is CrPC 482 quashing, 438 is
anticipatory bail), so scoring is by number. The IL-TUR **test** split is never indexed
(`iltur_export.assert_disjoint`). Tune on `--split dev`; report on `--split test`.
Every judgment is indexed; the 3,967 judgments `decontaminate.py` flags as near-duplicates
of a test case (and the dev-flagged ones when scoring dev) are masked only at scoring time.

## Results (official IL-TUR `lsi` protocol)

sklearn macro-F1 over the 100 label names on all 13,019 test cases; thresholds and fusion
weights fitted on dev only. Published: LeSICiN 28.08, InLegalBERT 26.23, GPT-4 0-shot 23.99.

| System | Test macro-F1 | Without near-dup cases |
|---|---|---|
| InLegalBERT chunked classifier | 38.35 | 37.43 |
| Classifier + 0.25 x precedent kNN (train-only memory) | 38.48 | 37.49 |
| Precedent kNN, train+dev memory / train-only / strict near-dup masking | 31.3 / 29.6 / 28.1 | |
| Judgment-passage section votes alone | 16.15 | |

The first two rows are confirmed by the leaderboard's own `evaluate_lsi`
(`verify_iltur_leaderboard.py classifier [precedent_trainmem:0.25]`). "Without near-dup
cases" drops the 891 test cases that share >=50% of their text with a train case.

The judgment votes (dev-fitted weight 0) and the fine-tuned label reranker (dev candidate
MRR 0.455 -> 0.606, still below precedent's 0.687; no fusion gain on 1,536 dev cases) add
nothing to the benchmark. Both stay available: the judgment index serves chat.
