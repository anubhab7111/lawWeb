#!/usr/bin/env python3
"""
Run script for the legal chatbot server.
"""

import asyncio

import uvicorn
from app.config import get_settings
from app.tools.offline_index import ensure_indices_built


def main():
    """Run the FastAPI server."""
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
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
