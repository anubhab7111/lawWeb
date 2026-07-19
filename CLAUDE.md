# CLAUDE.md

Guidance for working on this repository.

## Python environment — important

**Always use the conda env `legal_chatbot_env`** for anything Python (running the server, scripts, installing deps):

```bash
conda activate legal_chatbot_env
# or directly: /home/ushtro/.conda/envs/legal_chatbot_env/bin/python
```

Do **not** use `server/myenv` (an empty stray venv) or the system Python — neither has the dependencies. When adding a dependency, install it into `legal_chatbot_env` **and** add it to `server/requirements.txt`.

## Layout

- `server/` — the single FastAPI backend (port 8000). `app/main.py` wires routers from `app/routers/` (auth, lawyers, bookings, chat); `app/chatbot.py` is the LangGraph chatbot; `app/tools/` holds RAG systems and document pipeline; `app/db/` holds SQLModel models, `schema.sql`, and `init_db`.
- `client/` — React + Vite frontend (port 5173). API base URL from `VITE_API_URL`, default `http://localhost:8000/api`.
- There is no Node backend. JS/TS exists only in `client/`.
- `SYSTEM_DESIGN.md` and `CHATBOT_ARCHITECTURE.md` are **outdated** (they describe the old Express/Mongo/mistral setup) — do not use them as reference; trust the code.

## Running

```bash
cd server && python run.py        # backend; DB init first time: python -m app.db.init_db
cd client && npm run dev          # frontend
```

- Ollama must be running with `qwen3:14b` (model names in `app/config.py`).
- Run uvicorn with a **single worker**: hardware budget is ~15GB RAM / 4GB VRAM, and each worker would load its own copy of the embedding models. Keep embeddings/reranker off small GPUs — the VRAM is worth more to Ollama.

## Database

Local PostgreSQL, db/role `lawweb`/`lawweb`, connection via `DATABASE_URL` in `server/.env` (never commit `.env`). Schema is plain SQL in `server/app/db/schema.sql`; `python -m app.db.init_db` is idempotent (creates tables if missing, seeds 5 demo lawyers with legacy text ids `'1'..'5'`). To reset: `dropdb`/`createdb` as postgres (`psql -U postgres -h 127.0.0.1` works without sudo), then rerun `init_db`.

## API compatibility rules (client depends on these)

- Auth/lawyers/bookings errors return `{"message": "..."}` bodies — not FastAPI's default `{"detail"}`.
- Lawyer and booking JSON uses **camelCase** keys (`hourlyRate`, `successRate`, `userId`, `transactionId`, …) via the models' `to_dict()` helpers.
- `GET /api/bookings/client_token` returns **plain text**, not JSON.
- Chat endpoint paths under `/api/chat` are fixed — the client calls them directly.
- JWT secret/algorithm (HS256) and bcrypt hash format must stay compatible with existing users and tokens.

## RAG

- Corpus: `server/app/data/bare_acts/<domain>/*.pdf`; indices: `server/app/data/faiss_index/<domain>/`.
- Rebuild after corpus changes: `python rebuild_rag_indices.py --all` (run from `server/`; scripts chdir themselves because data paths resolve relative to CWD).
- Accuracy sweep: `python test_chatbot.py` (slow; needs Ollama).

## Payments

Braintree **sandbox** via the Python SDK; credentials only from `server/.env`. Headless checkout test: nonce `fake-valid-nonce`.
