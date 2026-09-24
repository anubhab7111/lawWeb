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
Court corpus (38,338 judgments, ~1.07M chunks) is prepared data; no retrieval index is
built from it (a judgment-passage index was tried and removed: see below).

## Regenerating (from `server/`, conda env `legal_chatbot_env`)

| Step | Command |
|---|---|
| SC judgments (verified, resumable) | `python -m app.ingest.download_sc` |
| Extract + clean + chunk judgments | `python -m app.ingest.process_judgments` |
| IL-TUR export (needs HF token, gated) | `python -m app.ingest.iltur_export` |
| Precedent index (train+dev) | `EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_precedent_index` |
| Section maps (IPC->BNS, CrPC->BNSS) | `python -m app.ingest.section_maps` |
| Decontaminate judgments vs IL-TUR test/dev | `python -m app.ingest.decontaminate [--split dev]` |
| Statute + case-law indices | `python rebuild_rag_indices.py --all` |
| IL-TUR retrieval eval | `python eval_iltur_retrieval.py --split test --tag <name>` |
| Cache retrieval output for offline tuning | `python tune_iltur_retrieval.py cache ...` |

Run GPU builds with Ollama idle (see hardware constraints in `CLAUDE.md`).

## Evaluation validity

IL-TUR `lsi` labels are **bare section numbers**. The dataset attaches IPC text to each
one, but the cases beneath mix IPC and CrPC (label 482 is CrPC 482 quashing, 438 is
anticipatory bail), so scoring is by number. The IL-TUR **test** split is never indexed
(`iltur_export.assert_disjoint`). Tune on `--split dev`; report on `--split test`.
Judgments sharing near-verbatim text with a test/dev case can be found with
`decontaminate.py` before any use of the judgment corpus for retrieval.

## Tried and removed

A hybrid (dense + FTS5) index over IPC/CrPC-citing Supreme Court passages, voting for
the sections their judgments cite, reached Hit@5 ~0.75 alone but added nothing to the
tuned fusion (best weight 0) and got worse when widened to CrPC-citing judgments
(procedural passages match any police narrative). Removed per the keep-only-if-it-helps
rule; it is recoverable from git history (`judgment_rag.py`, `build_judgment_index.py`).
