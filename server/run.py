#!/usr/bin/env python3
"""
Run script for the legal chatbot server.
"""

import asyncio
import sys

import uvicorn
from app.config import get_settings
from app.tools.offline_index import ensure_indices_built


def main():
    """Run the FastAPI server."""
    # Python block-buffers stdout when it isn't a terminal (e.g. redirected
    # to a log file, as any real deployment/demo run does) — the warmup and
    # indexing progress prints below would then sit invisible in a pipe
    # buffer for minutes before flushing, making a server that's working
    # normally look hung to anyone tailing the log. Force line buffering so
    # `python run.py > server.log 2>&1 &` shows progress in real time.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # Python <3.7 fallback: no-op, buffering stays default

    settings = get_settings()

    # Offline indexing step: build/refresh any missing or stale RAG index
    # with the embedding model on the GPU (safe here — Ollama hasn't loaded
    # the LLM into VRAM yet), then free that GPU memory before the server
    # starts. A no-op, disk-only check if every index is already current.
    asyncio.run(ensure_indices_built())

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Legal Chatbot Server                               ║
║                                                              ║
║  Features:                                                   ║
║  • Document Analysis - Upload and analyze legal documents    ║
║  • Crime Reporting - Get guidance on reporting crimes        ║
║  • Lawyer Finder - Search for lawyers by specialization      ║
║                                                              ║
║  API Docs: http://{settings.host}:{settings.port}/docs       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
