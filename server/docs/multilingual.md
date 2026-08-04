# Multilingual Support

LawWeb lets users ask legal questions in any major Indian language while the
knowledge base, retrieval pipeline and Qwen reasoning stay **English**. A thin
multilingual layer translates the query in and the answer out; conversation
memory is stored canonically in English.

## Architecture

```
User query (e.g. Hindi)
    │
    ▼
Language detection (fastText lid.176.bin)  ──low confidence──▶ assume English
    │ reliable non-English
    ▼
Mask legal entities  →  translate query → English (IndicTrans2 indic→en)  →  unmask
    │
    ▼
EXISTING pipeline, unchanged:
    classify_intent (embedding router) → hybrid retrieval (BM25 + BGE-M3 dense
    + cross-encoder rerank) → citations → Qwen (English reasoning)
    │
    ▼
English answer
    │
    ▼
Mask legal entities  →  translate answer → user language (IndicTrans2 en→indic)  →  unmask
    │
    ▼
User sees the answer in their language.  DB/memory keep the English canonical.
```

Key property: **the RAG pipeline never changes.** Translation happens only
*before* `classify_intent` and *after* the answer is generated. English requests
are a zero-cost no-op — no detection/translation model is loaded.

## Request flow & where it hooks in

| Stage | Code |
| --- | --- |
| Detect + translate query in | `app/multilingual/pipeline.py::preprocess_query`, called at the top of `LegalChatbot.chat` / `stream_chat` (`app/chatbot.py`) |
| Translate answer out | `pipeline.py::postprocess_response`, called after `graph.ainvoke` / after streaming completes |
| Persist canonical English | `app/routers/chat.py::_persist_turn` — `content` = English, `content_display` + `language` = user's original |
| Re-render history in-language | `GET /api/chat/session/{id}/history` returns `content_display` when present |

### Streaming (hybrid behaviour)
The SSE endpoint (`POST /api/chat/stream`) streams token-by-token **only for
English**. A non-English answer can be translated only after the full English
text exists, so those requests suppress per-token output and emit a single
translated message in the final `done` event. This is deliberate — see the
`stream_chat` docstring.

## Supported languages

fastText detects the language; IndicTrans2 translates it. Mapping lives in
`app/multilingual/lang_map.py` (ISO-639-1 → IndicTrans2 tag):

Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi,
Urdu, Odia, Assamese, Sanskrit, Konkani, Kashmiri, Manipuri (Meitei), Nepali,
Sindhi, Maithili, Bhojpuri, Santali, Dogri, Bodo.

A detected language outside this set (or below the confidence threshold) is
handled as English — no translation, pipeline runs as-is.

## Legal-reference preservation

`app/multilingual/entity_guard.py` masks section/article numbers, statute names,
Act acronyms (IPC, BNS, CrPC…), reported citations (`AIR 1973 SC 1461`,
`(2019) 3 SCC 12`), case names (`X v. Y`) and dates with opaque sentinel tokens
before translation, then restores them — so citations are never translated or
transliterated.

## Configuration

All settings live in `app/config.py` (env overrides by field name):

| Setting | Env | Default | Purpose |
| --- | --- | --- | --- |
| `multilingual_enabled` | `MULTILINGUAL_ENABLED` | `true` | Master switch; `false` = English-only, no models load |
| `language_detector` | `LANGUAGE_DETECTOR` | `fasttext` | Detector backend |
| `lang_detect_model_path` | `LANG_DETECT_MODEL_PATH` | `app/data/models/lid.176.bin` | fastText model file |
| `lang_detect_min_confidence` | `LANG_DETECT_MIN_CONFIDENCE` | `0.55` | Below this → assume English |
| `default_language` | `DEFAULT_LANGUAGE` | `en` | Fallback language |
| `translation_model_indic_en` | `TRANSLATION_MODEL_INDIC_EN` | `adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M` | query→English (CTranslate2) |
| `translation_model_en_indic` | `TRANSLATION_MODEL_EN_INDIC` | `adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M` | answer→user lang (CTranslate2) |
| `translation_device` | `TRANSLATION_DEVICE` | `cpu` | `auto`/`cuda`/`cpu` |
| `translation_cache` | `TRANSLATION_CACHE` | `true` | TTLCache on translations |
| `embedding_model` | `EMBEDDING_MODEL` | `BAAI/bge-m3` | Shared dense embedding model |
| `embedding_query_instruction` | `EMBEDDING_QUERY_INSTRUCTION` | `` (blank) | Must stay blank for BGE-M3 |

## Setup

### 1. Install dependencies
```bash
conda activate legal_chatbot_env
pip install -r requirements.txt   # adds fasttext-wheel, ctranslate2, IndicTransToolkit, sentencepiece
```
`fasttext-wheel` has no prebuilt wheel for very new interpreters (e.g. Python
3.14) and its sdist fails to compile on GCC 13+ because it relies on transitive
`<cstdint>` includes that newer GCC dropped. If the build fails, add
`#include <cstdint>` to the top of the `src/*.h` and `src/*.cc` files in the
sdist and `pip install` the patched source (one-time; the built wheel is cached).

### 2. Download the fastText language-ID model (~126 MB, one-time)
```bash
mkdir -p app/data/models
curl -L -o app/data/models/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

### 3. Translation models (CTranslate2 IndicTrans2)
Downloaded automatically from the Hugging Face Hub on first non-English request
(cached under `~/.cache/huggingface`). These are **non-gated** CTranslate2
conversions of Raj Dabre's rotary IndicTrans2 distilled 200M models, so no HF
license acceptance or token is required. To pre-fetch:
`huggingface-cli download adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M`.

**Why CTranslate2 rather than transformers:** this stack runs transformers 5.x
(required by sentence-transformers / BGE-M3 / the reranker), but IndicTrans2's
official HuggingFace models load via `trust_remote_code` modeling + tokenizer
code written for transformers ~4.x, which breaks on 5.x in several places
(removed `transformers.onnx`, moved `PreTrainedTokenizerBase`, stricter
`SpecialTokensMixin.__init__`). CTranslate2 runs the model with its own engine
and needs no transformers modeling code, so translation stays compatible without
downgrading transformers for the whole app. `IndicProcessor` (normalization,
language-tagging, transliteration) and SentencePiece handle tokenisation; the
only transformers shim needed is re-exporting `PreTrainedTokenizerBase` so
`IndicTransToolkit` imports (see `translation.py`).

### 4. Re-index with BGE-M3 (required — vectors are model-specific)
The index self-invalidates on model change (the embedding model name is stored
in `meta.pkl`), but force a clean rebuild to be safe:
```bash
rm -rf app/data/faiss_index/unified app/data/faiss_index/case_law
EMBEDDINGS_DEVICE=cuda python rebuild_rag_indices.py --all   # unified + case_law
```
Re-embed the pgvector consumers (they also use the shared model):
```bash
EMBEDDINGS_DEVICE=cuda python backfill_lawyer_embeddings.py --all
EMBEDDINGS_DEVICE=cuda python backfill_vault_embeddings.py --all
```
Existing databases pick up the two new `chat_messages` columns automatically via
`app/db/migrations.py` on next startup.

## Performance considerations

- **English is free.** Detection returns immediately for the default language;
  no translation model loads. `multilingual_enabled=false` disables the layer
  entirely.
- **Lazy loading.** fastText and both CTranslate2 IndicTrans2 models load on the
  first non-English request (singleton + lock), so their RAM cost (~0.5 GB
  /direction) is only paid when actually used. On the 15 GB / 4 GB-VRAM target,
  translation runs on **CPU** so the GPU stays reserved for Ollama's LLM.
- **Caching.** Identical (direction, text) translations are served from a
  TTLCache (`cache_ttl_seconds`).
- **Async.** The blocking `translate_batch` is offloaded to a thread; the request
  path stays async.
- **Distilled models.** The 200M distilled IndicTrans2 checkpoints are the
  default for the RAM budget; CTranslate2 int8 keeps them light and fast on CPU.
- **Fallbacks never break the chat.** If detection or translation fails, the
  layer logs and continues in English.
