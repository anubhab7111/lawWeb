# LawWeb — Law Education Platform with AI Legal Assistant

A full-stack platform for Indian law: an AI legal chatbot grounded in bare acts via RAG, legal document analysis and statutory validation, crime-reporting guidance, and a lawyer directory with bookings and sandbox payments.

**Stack:** React + Vite client · single Python FastAPI backend · local PostgreSQL · local LLMs via Ollama · FAISS vector search. Everything runs locally; no cloud services except the Braintree sandbox.

## Features

- **Legal chatbot** (`/api/chat`, streaming SSE) — LangGraph workflow with intent routing into domain RAG tools (criminal / civil / constitutional) built over Indian bare acts (IPC, BNS, BNSS, BSA, Constitution, and ~40 more in `server/app/data/bare_acts/`), plus Indian Kanoon case-law lookup.
- **Document analysis & validation** — upload PDF/DOCX/images (OCR via Tesseract); a 3-layer pipeline classifies the document, checks statutory requirements, and flags legal defects.
- **Crime reporting guidance** — structured steps for reporting, by detected crime type.
- **Lawyer directory & bookings** — Postgres-backed lawyer listing/recommendation, JWT auth, and Braintree (sandbox) checkout.

## Prerequisites

- **Conda env** `legal_chatbot_env` (Python 3.14) — the project's only supported Python environment.
- **PostgreSQL** running locally (one-time setup below).
- **Ollama** with `qwen3:14b` pulled (the answering model; see `server/app/config.py`).
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
python -m app.db.init_db           # creates tables from app/db/schema.sql + seeds demo lawyers (idempotent)
python run.py                      # FastAPI on http://localhost:8000 (API docs at /docs)

# Client
cd client
npm install
npm run dev                        # http://localhost:5173 (API base overridable via VITE_API_URL)
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
python rebuild_rag_indices.py --all      # or --domain criminal|civil|constitutional
```

## Testing

```bash
cd server
python test_chatbot.py    # accuracy sweep over domain prompts; needs Ollama running, slow
```

See `CLAUDE.md` for development conventions.
