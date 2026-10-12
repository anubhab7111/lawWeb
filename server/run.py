#!/usr/bin/env python3
"""Run script for the legal chatbot server."""

import asyncio
import sys

import uvicorn
from app.config import get_settings
from app.tools.offline_index import ensure_indices_built


def main():
    """Run the FastAPI server."""


    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    settings = get_settings()


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
