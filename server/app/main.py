"""FastAPI backend for the legal platform."""

import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.checkpointing import close_checkpointer, init_checkpointer
from app.config import get_settings
from app.deps.errors import MessageHTTPException
from app.logging_config import configure_logging
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

configure_logging(get_settings().log_level)


async def _warmup() -> None:
    """Pre-load everything the chatbot lazily initializes on first use — BGE-M3 embeddings, the reranker, the unified + case-law FAISS/BM25 indices, the fastText language detector,..."""
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


def _ensure_chat_schema() -> None:
    """The chat_messages.trace column postdates already-deployed databases and migrations otherwise run only via `python -m app.db.init_db`."""
    try:
        from app.db.engine import get_engine
        from app.db.migrations import ensure_chat_messages_trace_column

        ensure_chat_messages_trace_column(get_engine())
    except Exception:
        traceback.print_exc()
        print("[Startup] could not ensure chat_messages.trace; run `python -m app.db.init_db`")


@asynccontextmanager
async def lifespan(app: FastAPI):


    _ensure_chat_schema()

    await init_checkpointer()
    scheduler = get_scheduler()
    register_jobs(scheduler)
    scheduler.start()
    warmup_task = asyncio.create_task(_warmup())
    try:
        yield
    finally:
        warmup_task.cancel()
        scheduler.shutdown(wait=False)
        await close_checkpointer()


app = FastAPI(
    title="Legal Platform API",
    description="AI-powered legal assistant with document analysis, crime reporting guidance, lawyer search, auth, and bookings.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/config")
async def public_config():
    """Non-secret settings the client needs to render prices honestly."""
    import json

    from app.services.fcm_client import push_configured

    settings = get_settings()
    firebase = None
    if push_configured() and settings.firebase_web_config_json and settings.firebase_vapid_key:
        try:
            firebase = {
                "config": json.loads(settings.firebase_web_config_json),
                "vapidKey": settings.firebase_vapid_key,
            }
        except ValueError:
            firebase = None
    from app.services.calendar_oauth import configured_providers

    return {
        "currency": settings.currency,
        "firebase": firebase,
        "calendarProviders": configured_providers(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.exception_handler(MessageHTTPException)
async def message_http_exception_handler(request, exc: MessageHTTPException):
    """Scoped to MessageHTTPException only (raised by app.deps.auth and the new feature routers) so existing HTTPException usage elsewhere (e.g."""
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
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


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.reload
    )
