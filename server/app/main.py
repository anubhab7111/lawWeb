"""
FastAPI backend for the legal platform.
Single Python backend: chatbot/RAG plus auth, lawyers, and bookings
(previously served by the Express server).
"""

import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.deps.errors import MessageHTTPException
from app.routers import (
    auth,
    bare_acts,
    bookings,
    calendar,
    cases,
    cause_list,
    chat,
    lawyers,
    notifications,
    similar_cases,
    vault,
)
from app.scheduler import get_scheduler, register_jobs


async def _warmup() -> None:
    """Pre-load everything the chatbot lazily initializes on first use —
    BGE-M3 embeddings, the reranker, the unified + case-law FAISS/BM25
    indices, the fastText language detector, and the Ollama LLM (into VRAM,
    kept resident via keep_alive) — so the first real user query isn't the
    one paying for it. Measured cold-start cost without this: 10+ minutes.

    Runs as a background task so it never delays uvicorn accepting
    connections, and must never raise: any failure here just means the
    first request falls back to the existing per-component lazy-init path.

    Uses print(), not the logging module — nothing in this app configures a
    logging handler/level (uvicorn only configures its own loggers), so
    logger.info()/logger.exception() calls here would be silently dropped.
    print() is what every other diagnostic line in this codebase uses.
    """
    try:
        from app.chatbot import get_fast_llm, get_llm, invoke_llm_safely
        from app.intent_classifier import classify_intent_embedding
        from app.multilingual.language_detection import detect_language
        from app.tools.case_law_rag import get_case_law_rag_system
        from app.tools.unified_legal_rag import get_unified_rag_system

        print("[Warmup] starting background warmup...")
        await get_unified_rag_system().initialize()
        await get_case_law_rag_system().initialize()
        await classify_intent_embedding("What is the punishment for theft?", False)
        await detect_language("test")
        await invoke_llm_safely(get_llm(), "Reply with OK.", stream=False)
        await invoke_llm_safely(get_fast_llm(), "Reply with OK.", stream=False)
        print("[Warmup] complete.")
    except Exception:
        print("[Warmup] failed; components will lazy-init on first request")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First-ever startup hook in this app. Scheduler jobs are durable
    # (SQLAlchemy jobstore) so restarts don't need to recreate DB state, but
    # add_job(..., replace_existing=True) inside register_jobs() still runs
    # every boot to pick up code changes to job schedules.
    scheduler = get_scheduler()
    register_jobs(scheduler)
    scheduler.start()
    warmup_task = asyncio.create_task(_warmup())
    try:
        yield
    finally:
        warmup_task.cancel()
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Legal Platform API",
    description="AI-powered legal assistant with document analysis, crime reporting guidance, lawyer search, auth, and bookings.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend integration. Origins are an explicit allowlist
# (not "*") because allow_credentials=True combined with a wildcard would let
# any origin make credentialed requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.include_router(auth.router)
app.include_router(lawyers.router)
app.include_router(bookings.router)
app.include_router(chat.router)
app.include_router(bare_acts.router)
app.include_router(similar_cases.router)
app.include_router(cases.router)
app.include_router(notifications.router)
app.include_router(cause_list.router)
app.include_router(vault.router)
app.include_router(calendar.router)


@app.get("/")
async def root():
    """Serve the frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(MessageHTTPException)
async def message_http_exception_handler(request, exc: MessageHTTPException):
    """Scoped to MessageHTTPException only (raised by app.deps.auth and the
    new feature routers) so existing HTTPException usage elsewhere (e.g.
    app.routers.chat) keeps its current {"detail": ...} response shape."""
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors.

    FastAPI/Starlette places a bare-Exception handler in
    ServerErrorMiddleware, which sits OUTSIDE CORSMiddleware — so a response
    built here never gets CORS headers from that middleware, and the browser
    reports every unhandled 500 as an opaque "Failed to fetch"/CORS error
    instead of surfacing the actual message. Attach the header manually
    (only for an origin already on the allowlist), and use the
    {"message": ...} shape the client's error handling expects everywhere
    else (see app.deps.errors.MessageHTTPException).
    """
    print(f"Unhandled exception on {request.method} {request.url.path}")
    traceback.print_exc()
    settings = get_settings()
    response = JSONResponse(
        status_code=500,
        content={
            "message": str(exc) if settings.debug else "An unexpected error occurred."
        },
    )
    origin = request.headers.get("origin")
    if origin and origin in settings.cors_origins_list:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# ============================================================================
# Main Entry Point
# ============================================================================


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.reload
    )
