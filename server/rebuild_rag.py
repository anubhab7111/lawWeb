import asyncio
import os
import sys

# Ensure server/ directory is in sys.path and is the CWD, since
# BaseLegalRAGSystem resolves its data_dir ("app/data") relative to CWD.
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SERVER_DIR)
os.chdir(_SERVER_DIR)


async def rebuild_all():
    from app.tools import get_unified_rag_system

    system = get_unified_rag_system()

    print("\n--- Starting Unified RAG Index Rebuild (FAISS + BM25 hybrid) ---")

    ok = await system.initialize()
    if ok:
        print(f"SUCCESS: unified index ready ({len(system._chunks)} chunks)")
    else:
        print("FAILED to build unified index")


if __name__ == "__main__":
    asyncio.run(rebuild_all())
