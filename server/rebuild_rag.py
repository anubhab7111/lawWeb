import asyncio
import os
import sys

# Ensure server/ directory is in sys.path and is the CWD, since
# BaseLegalRAGSystem resolves its data_dir ("app/data") relative to CWD.
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SERVER_DIR)
os.chdir(_SERVER_DIR)


async def rebuild_all():
    from app.tools import (
        get_criminal_rag_system,
        get_civil_rag_system,
        get_constitutional_rag_system,
    )

    systems = [
        get_criminal_rag_system(),
        get_civil_rag_system(),
        get_constitutional_rag_system(),
    ]

    print("\n--- Starting Parallel RAG Index Rebuild (FAISS semantic search) ---")

    # Initialize all systems in parallel to trigger build
    tasks = [sys.initialize() for sys in systems]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failed = 0
    for i, res in enumerate(results):
        name = systems[i].domain_name
        if isinstance(res, Exception) or not res:
            failed += 1
            print(f"FAILED to build {name}: {res}")
        else:
            print(f"SUCCESS: {name} index ready ({len(systems[i]._chunks)} chunks)")

    if failed:
        print(f"\n--- {len(systems) - failed}/{len(systems)} domains indexed successfully ---")
    else:
        print(f"\n--- All {len(systems)} domains indexed successfully ---")


if __name__ == "__main__":
    asyncio.run(rebuild_all())
