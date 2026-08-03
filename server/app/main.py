"""
FastAPI backend for the legal platform.
Single Python backend: chatbot/RAG plus auth, lawyers, and bookings
(previously served by the Express server).
"""

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First-ever startup hook in this app. Scheduler jobs are durable
    # (SQLAlchemy jobstore) so restarts don't need to recreate DB state, but
    # add_job(..., replace_existing=True) inside register_jobs() still runs
    # every boot to pick up code changes to job schedules.
    scheduler = get_scheduler()
    register_jobs(scheduler)
    scheduler.start()
    try:
        yield
    finally:
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
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": (
                str(exc) if get_settings().debug else "An error occurred"
            ),
        },
    )


# ============================================================================
# Main Entry Point
# ============================================================================


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
