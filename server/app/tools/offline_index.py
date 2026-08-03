"""
offline_index.py — the "offline indexing" startup step.

    Start indexing
        v
    Load embedding model on GPU
        v
    Embed PDFs / case law
        v
    Save FAISS index
        v
    Free GPU memory
        v
    Start LLM

Runs once, synchronously, before uvicorn starts serving (see run.py). Each
RAG system gets its own throwaway embeddings instance forced onto the GPU
(see BaseLegalRAGSystem.build_offline / CaseLawRAGSystem.build_offline) —
cheap and safe because at this point Ollama hasn't loaded the LLM into VRAM
yet. If an index is already up to date (see the content-fingerprint check
in _should_rebuild), its build is skipped entirely and the GPU is never
touched, so re-running this on every `python run.py` is normally a no-op.

The query-time embedding singleton (app.tools.base_legal_rag._get_shared_
embeddings) is untouched by this module and keeps its own device logic —
on small GPUs it stays on CPU so the LLM keeps the VRAM once it loads.
"""

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
    """Build/refresh every RAG index that's missing or stale, then return.

    Uses freshly-constructed system instances rather than the request-
    serving singletons (get_unified_rag_system()/get_case_law_rag_system())
    so the singletons stay untouched and lazily self-initialize (a cheap
    disk load, since the index is now fresh) on the server's first request.
    """
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
