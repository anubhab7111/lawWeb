# Data layout

Source and derived corpus data no longer live in this directory. They are on the
corpus drive, `LAWWEB_CORPUS_ROOT` (set in `server/.env`), and only the vector /
lexical **indices** stay in the repo, under `faiss_index/`.

What remains here: `lawyers.json` (seed fixture for `init_db`), `legal_ontology.json`
and `case_law_manifest.py` (runtime), `vault/` (runtime upload storage, gitignored),
and `faiss_index/` (indices).

Serving never needs the drive: `unified` / `case_law` / `precedent` / `judgments`
indices are self-contained (chunk text and metadata are inside them). If the drive
is unmounted the staleness check is skipped and the existing index is loaded; only
rebuilds and ingestion need it.

## Corpus drive

`python -m app.ingest.corpus_layout --verify` (from `server/`) checks it. Layout:

```
statutes/           bare_acts/<domain>/*.pdf, mappings/, rules/, guides/, notifications/, explanatory/
case_law/curated/   landmark judgments (JSON)
judgments/sc/       pdf/ text/ clean/ chunks/ metadata/ manifest/   (Supreme Court, 1950-2026)
iltur/lsi/          IL-TUR export (CC BY-NC-SA 4.0 - non-commercial)
indiankanoon/       API response cache
builds/             embedding shards, eval + QA reports, decontamination report
manifest/           file hashes, download ledgers
quarantine/         files rejected by a quality gate, each with a reason
```

Layers flow one way: raw PDF -> `text/` -> `clean/` -> `chunks/` -> repo index.

## Regenerating (from `server/`, conda env `legal_chatbot_env`)

| Step | Command |
|---|---|
| SC judgments (verified, resumable) | `python -m app.ingest.download_sc` |
| Extract + clean + chunk judgments | `python -m app.ingest.process_judgments` |
| IL-TUR export (needs HF token, gated) | `python -m app.ingest.iltur_export` |
| Precedent index (train+dev) | `EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_precedent_index` |
| Decontaminate judgments vs IL-TUR test | `python -m app.ingest.decontaminate` |
| Judgment index | `EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_judgment_index` |
| Statute + case-law indices | `python rebuild_rag_indices.py --all` |
| IL-TUR retrieval eval | `python eval_iltur_retrieval.py --split test --tag <name>` |

Run GPU builds with Ollama idle (see hardware constraints in `CLAUDE.md`).

## Evaluation validity

IL-TUR `lsi` labels are IPC section numbers. The IL-TUR **test** split is never
indexed (`iltur_export.assert_disjoint`), and judgments sharing near-verbatim text
with a test case are excluded from the judgment index (`decontaminate.py`).
Tune on `--split dev`; report on `--split test`.
