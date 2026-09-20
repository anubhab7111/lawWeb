# LawWeb — Law Education Platform with AI Legal Assistant

A full-stack platform for Indian law: an AI legal chatbot grounded in bare acts via RAG, legal document analysis and statutory validation, crime-reporting guidance, and a lawyer directory with bookings and sandbox payments.

**Stack:** React + Vite client · single Python FastAPI backend · local PostgreSQL · local LLMs via Ollama · FAISS vector search. Everything runs locally; no cloud services except the Braintree sandbox.

## Features

- **Legal chatbot** (`/api/chat`, streaming SSE) — LangGraph workflow with intent routing into domain RAG tools (criminal / civil / constitutional) built over Indian bare acts (IPC, BNS, BNSS, BSA, Constitution, and ~40 more in `server/app/data/bare_acts/`), plus Indian Kanoon case-law lookup.
- **Document analysis & validation** — upload PDF/DOCX/images (OCR via Tesseract); a 3-layer pipeline classifies the document, checks statutory requirements, and flags legal defects.
- **Crime reporting guidance** — structured steps for reporting, by detected crime type.
- **Lawyer directory & bookings** — ~1,000 Indian lawyers (server-side filtering/paging, semantic recommendation), JWT auth, appointment slots, and idempotent Braintree (sandbox) checkout. Prices display in the charged currency (`CURRENCY`, default USD).
- **Legal Document Vault** — per-user storage with AI search, sharing by email (view/edit), download, and re-indexing.
- **My Cases, Cause List, Calendar, Notifications** — case tracking with hearing reminders, in-app/email/browser-push notifications (FCM), and optional Google/Outlook calendar sync.

Optional integrations (all off unless configured in `server/.env`): `FCM_SERVICE_ACCOUNT_JSON` + `FIREBASE_WEB_CONFIG_JSON` + `FIREBASE_VAPID_KEY` (push), `GOOGLE_CLIENT_ID/SECRET` and `MS_CLIENT_ID/SECRET` (calendar sync; redirect URI `<PUBLIC_API_URL>/api/calendar/sync/<google|outlook>/callback`), `SMTP_*` (email), `R2_*` (vault storage).

## Prerequisites

- **Conda env** `legal_chatbot_env` (Python 3.14) — the project's only supported Python environment.
- **PostgreSQL** running locally (one-time setup below).
- **Ollama** with `qwen3:4b` pulled (the default answering model; change `LLM_MODEL` / `FAST_LLM_MODEL` in `server/.env`, and set `LLM_THINKING=false` for models that don't emit a thinking block).
- **Tesseract + Poppler** for OCR (`pytesseract`, `pdf2image`).
- **Node 18+** for the client only.

### One-time PostgreSQL setup (Arch Linux)

```bash
sudo pacman -S --needed postgresql
sudo -u postgres initdb --locale=en_US.UTF-8 -E UTF8 -D /var/lib/postgres/data   # skip if already initialized
sudo systemctl enable --now postgresql
sudo -u postgres psql -c "CREATE ROLE lawweb LOGIN PASSWORD 'lawweb' CREATEDB;"
sudo -u postgres createdb -O lawweb lawweb
```

## Setup & Run

Create `server/.env` with `DATABASE_URL="postgresql://lawweb:lawweb@localhost:5432/lawweb?schema=public"`, `JWT_SECRET`, and your `BRAINTREE_MERCHANT_ID` / `BRAINTREE_PUBLIC_KEY` / `BRAINTREE_PRIVATE_KEY` sandbox keys.

```bash
# Backend
conda activate legal_chatbot_env
cd server
pip install -r requirements.txt
python -m app.db.init_db           # tables + the Indian lawyer directory (app/data/lawyers.json) + bio embeddings (idempotent; --skip-embeddings to skip)
python run.py                      # FastAPI on http://localhost:8000 (API docs at /docs)

# Client
cd client
npm install
npm run dev                        # http://localhost:3000 (API base overridable via VITE_API_URL)
npm run typecheck                  # tsc --noEmit
```

## API overview

| Prefix | Purpose |
|---|---|
| `/api/auth` | register / login / me (JWT, bcrypt) |
| `/api/lawyers` | list, detail, recommend (Postgres) |
| `/api/bookings` | Braintree client token, checkout, user bookings |
| `/api/chat` | chat (+ `/stream` SSE), document upload/analyze/validate, crime-report, find-lawyer, sessions |

## RAG indices

Prebuilt FAISS indices live in `server/app/data/faiss_index/<domain>/`. After changing the corpus, rebuild with:

```bash
cd server
python rebuild_rag_indices.py --all      # or --domain unified | case_law
```

## Testing

```bash
cd server
python tests/test_chatbot.py    # accuracy sweep over domain prompts; needs Ollama running, slow
```

Fast, Ollama-free unit tests: `cd server && python -m pytest tests/unit` (DB-backed tests use a scratch database — `createdb lawweb_scratch` with the `vector` extension, or set `TEST_DATABASE_URL`).

See `CLAUDE.md` for development conventions.
