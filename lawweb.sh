#!/usr/bin/env bash
# Set up and start LawWeb (FastAPI backend + Vite client).
#
#   ./lawweb.sh          setup, then start
#   ./lawweb.sh setup    install dependencies, create .env files, prepare the database, pull the LLM
#   ./lawweb.sh start    start Ollama (if needed), the backend and the client; Ctrl-C stops them
#
# Setup is safe to rerun: it never overwrites an existing .env and skips steps that are already done.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$ROOT/server"
CLIENT="$ROOT/client"
CONDA_ENV="${LAWWEB_CONDA_ENV:-legal_chatbot_env}"
PYTHON_VERSION="3.14"
BACKEND_WAIT_SECONDS="${LAWWEB_BACKEND_WAIT:-600}"
OLLAMA_LOG="${TMPDIR:-/tmp}/lawweb-ollama.log"

PY=""
PIDS=()

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

find_python() {
    [ -n "$PY" ] && return 0
    have conda || die "conda not found. Install Miniconda, then rerun."
    PY="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    [ -n "$PY" ] && [ -x "$PY" ]
}

# Print one value from the app's own settings, so .env quoting and defaults are handled by the app.
setting() {
    (cd "$SERVER" && "$PY" -c "from app.config import get_settings; print(get_settings().$1)")
}

db_query() {
    (cd "$SERVER" && "$PY" - "$1" <<'EOF'
import sys
from app.db.engine import get_engine

with get_engine().connect() as conn:
    row = conn.exec_driver_sql(sys.argv[1]).first()
    conn.commit()
    print(row[0] if row else "")
EOF
    )
}

check_tools() {
    info "Checking prerequisites"
    have conda || die "conda not found. Install Miniconda, then rerun."
    have node && have npm || die "Node.js 18+ and npm are required for the client."
    local node_major
    node_major="$(node -p 'process.versions.node.split(".")[0]')"
    [ "$node_major" -ge 18 ] || die "Node.js 18+ is required (found $(node -v))."
    have curl || die "curl is required."
    have ollama || die "Ollama not found. Install it from https://ollama.com/download, then rerun."
    have tesseract || warn "tesseract not found: OCR of scanned documents and images will not work."
    have pdftoppm || warn "poppler (pdftoppm) not found: OCR of scanned PDFs will not work."
}

setup_python() {
    if ! find_python; then
        info "Creating conda env $CONDA_ENV (Python $PYTHON_VERSION)"
        conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
        PY=""
        find_python || die "Could not find the python of conda env $CONDA_ENV."
    fi

    # The stamp lives inside the env, so a recreated env reinstalls.
    local stamp want
    stamp="$(dirname "$(dirname "$PY")")/.lawweb-requirements.sha256"
    want="$(sha256sum "$SERVER/requirements.txt" | cut -d' ' -f1)"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
        info "Python packages are up to date"
    else
        info "Installing Python packages (first run downloads several GB)"
        local reqs="$SERVER/requirements.txt"
        # A conda-installed faiss provides the same module; pip's faiss-gpu would overwrite its files and break it.
        if "$PY" -c 'import faiss' >/dev/null 2>&1; then
            reqs="$(mktemp)"
            grep -viE '^faiss' "$SERVER/requirements.txt" > "$reqs"
        fi
        "$PY" -m pip install -r "$reqs"
        echo "$want" > "$stamp"
    fi
}

setup_env_files() {
    if [ ! -f "$SERVER/.env" ]; then
        info "Creating server/.env from .env.example"
        cp "$SERVER/.env.example" "$SERVER/.env"
    fi
    if grep -qE '^JWT_SECRET=["'\'']?["'\'']?[[:space:]]*$' "$SERVER/.env"; then
        info "Generating JWT_SECRET in server/.env"
        local secret
        secret="$("$PY" -c 'import secrets; print(secrets.token_hex(32))')"
        sed -i -E "s|^JWT_SECRET=.*$|JWT_SECRET=$secret|" "$SERVER/.env"
    fi
    if [ -z "$(setting braintree_merchant_id)" ]; then
        warn "Braintree sandbox keys are empty in server/.env: booking checkout will not work until you add them."
    fi
    if [ ! -f "$CLIENT/.env" ]; then
        info "Creating client/.env from .env.example"
        cp "$CLIENT/.env.example" "$CLIENT/.env"
    fi
}

postgres_help() {
    cat >&2 <<'EOF'

One-time PostgreSQL setup (needs sudo; shown for Arch Linux, with the pgvector package):

    sudo pacman -S --needed postgresql pgvector
    sudo -u postgres initdb --locale=en_US.UTF-8 -E UTF8 -D /var/lib/postgres/data   # skip if already initialized
    sudo systemctl enable --now postgresql
    sudo -u postgres psql -c "CREATE ROLE lawweb LOGIN PASSWORD 'lawweb' CREATEDB;"
    sudo -u postgres createdb -O lawweb lawweb
    sudo -u postgres psql -d lawweb -c "CREATE EXTENSION IF NOT EXISTS vector;"

Then rerun this script.
EOF
}

setup_database() {
    info "Checking PostgreSQL"
    if ! db_query "SELECT 1" >/dev/null 2>&1; then
        if have pg_isready && ! pg_isready -q -h 127.0.0.1; then
            warn "PostgreSQL is not accepting connections on 127.0.0.1."
        else
            warn "Cannot connect with DATABASE_URL from server/.env (role or database missing?)."
        fi
        postgres_help
        exit 1
    fi
    # Creating the extension needs a superuser, so try as the app role and fall back to instructions.
    if [ -z "$(db_query "SELECT 1 FROM pg_extension WHERE extname = 'vector'")" ] &&
        ! db_query "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null 2>&1; then
        warn "The pgvector extension is not enabled in the database and the app role may not create it."
        postgres_help
        exit 1
    fi
    info "Creating tables and seeding lawyers (idempotent)"
    (cd "$SERVER" && "$PY" -m app.db.init_db)
}

setup_client() {
    local lock="$CLIENT/package-lock.json" installed="$CLIENT/node_modules/.package-lock.json"
    if [ -f "$installed" ] && [ ! "$lock" -nt "$installed" ]; then
        info "Client packages are up to date"
    else
        info "Installing client packages"
        (cd "$CLIENT" && npm install)
    fi
}

ollama_up() { curl -fsS -m 3 "$1/api/tags" >/dev/null 2>&1; }

ensure_ollama() {
    local url
    url="$(setting ollama_base_url)"
    ollama_up "$url" && return 0
    info "Starting Ollama"
    setsid ollama serve >"$OLLAMA_LOG" 2>&1 &
    PIDS+=("$!")
    for _ in $(seq 1 30); do
        ollama_up "$url" && return 0
        sleep 1
    done
    die "Ollama did not come up at $url (see $OLLAMA_LOG)."
}

setup_models() {
    ensure_ollama
    local model
    for model in $(printf '%s\n' "$(setting llm_model)" "$(setting fast_llm_model)" | sort -u); do
        if ollama show "$model" >/dev/null 2>&1; then
            info "Ollama model $model is present"
        else
            info "Pulling Ollama model $model"
            ollama pull "$model"
        fi
    done
    if [ ! -d "$SERVER/app/data/faiss_index/unified" ]; then
        warn "No search index in server/app/data/faiss_index/unified: the backend builds it on first start, which takes a long time."
    fi
}

cmd_setup() {
    check_tools
    setup_python
    setup_env_files
    setup_database
    setup_client
    setup_models
    info "Setup complete"
}

stop_all() {
    trap - INT TERM EXIT
    local pid
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 10); do
        [ -z "$(jobs -rp)" ] && return 0
        sleep 1
    done
    for pid in "${PIDS[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
    done
}

cmd_start() {
    find_python || die "Conda env $CONDA_ENV not found. Run: ./lawweb.sh setup"
    [ -f "$SERVER/.env" ] || die "server/.env not found. Run: ./lawweb.sh setup"
    [ -d "$CLIENT/node_modules" ] || die "Client packages not installed. Run: ./lawweb.sh setup"

    local port api
    port="$(setting port)"
    api="http://127.0.0.1:$port"
    if curl -fsS -m 3 "$api/health" >/dev/null 2>&1; then
        die "Something already answers on $api. Stop it first."
    fi

    ensure_ollama

    # Each service gets its own session so stop_all can signal its whole process group.
    info "Starting backend on $api"
    (cd "$SERVER" && exec setsid "$PY" run.py) &
    local backend=$!
    PIDS+=("$backend")

    local waited=0
    until curl -fsS -m 3 "$api/health" >/dev/null 2>&1; do
        kill -0 "$backend" 2>/dev/null || die "The backend exited during startup (see the output above)."
        [ "$waited" -lt "$BACKEND_WAIT_SECONDS" ] || die "The backend did not answer within ${BACKEND_WAIT_SECONDS}s."
        sleep 2
        waited=$((waited + 2))
    done
    info "Backend is up (API docs: $api/docs)"

    info "Starting client on http://localhost:3000"
    (cd "$CLIENT" && exec setsid npm run dev) &
    PIDS+=("$!")

    info "LawWeb is running. Press Ctrl-C to stop."
    wait -n || true
    die "A service exited; stopping the rest."
}

trap stop_all EXIT
trap 'exit 130' INT TERM

case "${1:-all}" in
setup) cmd_setup ;;
start) cmd_start ;;
all)
    cmd_setup
    cmd_start
    ;;
*) die "Usage: $0 [setup|start]" ;;
esac
