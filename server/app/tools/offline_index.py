"""offline_index.py — the "offline indexing" startup step."""

from __future__ import annotations

from app.tools.case_law_rag import CaseLawRAGSystem
from app.tools.unified_legal_rag import UnifiedLegalRAGSystem


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


async def ensure_indices_built() -> None:
    """Build/refresh every RAG index that's missing or stale, then return."""
    device = _pick_device()
    systems = [
        ("unified", UnifiedLegalRAGSystem()),
        ("case_law", CaseLawRAGSystem()),
    ]
    for label, system in systems:
        try:
            ok = await system.build_offline(device=device)
            if not ok:
                print(f"[startup] {label} index build failed — will retry lazily on first request.")
        except Exception as e:
            print(f"[startup] {label} offline build error: {e} — will retry lazily on first request.")
