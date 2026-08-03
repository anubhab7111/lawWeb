"""
Indexes a Vault document after upload: extract text (reusing the existing
DocumentExtractor — the same class chat's /upload and Similar Case Search
use), chunk it, embed each chunk with the *shared* embedding singleton (no
second model loaded, per the hardware budget), and write
vault_document_embeddings rows for pgvector search.

Run as a FastAPI BackgroundTasks callback from the upload endpoint
(fire-and-forget) — not a scheduled job, since it's a per-upload event.
"""

import asyncio

from sqlmodel import Session

from app.db.engine import get_engine
from app.db.models import VaultDocument, VaultDocumentEmbedding
from app.tools.base_legal_rag import _get_shared_embeddings

_CHUNK_CHARS = 1500
_CHUNK_OVERLAP = 200


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_CHARS
        chunks.append(text[start:end])
        start = end - _CHUNK_OVERLAP
        if start <= 0:
            break
    return [c for c in chunks if c.strip()]


async def index_vault_document(document_id: str, extracted_text: str) -> None:
    with Session(get_engine()) as session:
        document = session.get(VaultDocument, document_id)
        if document is None:
            return

        document.indexing_status = "indexing"
        document.extracted_text = extracted_text
        session.add(document)
        session.commit()

        try:
            chunks = _chunk_text(extracted_text)
            if not chunks:
                document.indexing_status = "ready"
                session.add(document)
                session.commit()
                return

            embeddings_model = await _get_shared_embeddings()
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(None, embeddings_model.embed_documents, chunks)

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                session.add(
                    VaultDocumentEmbedding(
                        vault_document_id=document.id,
                        chunk_index=i,
                        chunk_text=chunk,
                        embedding=vector,
                    )
                )

            document.indexing_status = "ready"
            session.add(document)
            session.commit()
        except Exception as e:
            print(f"[VaultIndexer] indexing failed for document {document_id}: {e}")
            document.indexing_status = "failed"
            session.add(document)
            session.commit()
