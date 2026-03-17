#!/usr/bin/env python3
"""
Run script for the legal chatbot server.
"""
import uvicorn
from app.config import get_settings


def main():
    """Run the FastAPI server."""
    settings = get_settings()

    print(
        f"""
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
    """
    )

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
