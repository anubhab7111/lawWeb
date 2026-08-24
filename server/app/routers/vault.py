"""
Legal Document Vault — secure storage (Cloudflare R2, local-disk fallback in
dev) plus AI-powered search over per-user documents via pgvector, not a new
vector DB or FAISS index (see the implementation plan's pgvector-vs-Qdrant
rationale: already-enabled extension, small per-user document sets, and
trivial SQL-side permission filtering).
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session, or_, select

from app.config import get_settings
from app.db.engine import get_session
from app.db.models import User, VaultDocument, VaultDocumentEmbedding, VaultDocumentPermission
from app.deps.auth import get_current_user
from app.deps.errors import MessageHTTPException
from app.deps.uploads import read_upload_within_limit
from app.services.notification_dispatch import send_notification
from app.services.object_storage import LocalDiskObjectStorage, get_object_storage
from app.services.vault_indexer import index_vault_document
from app.tools.base_legal_rag import _get_shared_embeddings
from app.tools.document_extractor import get_document_extractor

router = APIRouter(prefix="/api/vault", tags=["vault"])


class SearchRequest(BaseModel):
    query: str


class ShareRequest(BaseModel):
    sharedWithUserId: str
    permission: str = "view"


def _accessible_ids_subquery(session: Session, user_id: str):
    shared_doc_ids = session.exec(
        select(VaultDocumentPermission.vault_document_id).where(
            VaultDocumentPermission.shared_with_user_id == user_id
        )
    ).all()
    return list(shared_doc_ids)


def _get_accessible_document(session: Session, document_id: str, user: User) -> VaultDocument:
    document = session.get(VaultDocument, document_id)
    if document is None:
        raise MessageHTTPException(status_code=404, detail="Document not found")
    if document.user_id == user.id:
        return document
    shared = session.exec(
        select(VaultDocumentPermission).where(
            VaultDocumentPermission.vault_document_id == document_id,
            VaultDocumentPermission.shared_with_user_id == user.id,
        )
    ).first()
    if shared is None:
        raise MessageHTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: str = Form(..., alias="documentType"),
    related_case_id: Optional[str] = Form(default=None, alias="relatedCaseId"),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    max_size = get_settings().max_document_size_mb * 1024 * 1024
    file_bytes = await read_upload_within_limit(file, max_size)

    # Basename only: strip any directory components from the client-supplied
    # filename so it can't inject `..`/separators into the storage key (which
    # the local-disk fallback resolves against LOCAL_VAULT_DIR).
    safe_filename = Path(file.filename or "document").name or "document"
    object_key = f"{current_user.id}/{uuid.uuid4()}-{safe_filename}"

    storage = get_object_storage()
    storage.upload_object(object_key, file_bytes, file.content_type or "application/octet-stream")

    document = VaultDocument(
        user_id=current_user.id,
        title=title,
        document_type=document_type,
        object_key=object_key,
        file_size_bytes=len(file_bytes),
        mime_type=file.content_type or "application/octet-stream",
        related_case_id=related_case_id,
        indexing_status="pending",
    )
    session.add(document)
    session.commit()
    session.refresh(document)

    extractor = get_document_extractor()
    try:
        extracted_text, _doc_type = await extractor.extract_text(file_bytes, file.filename or "document.txt")
    except Exception as e:
        print(f"[Vault] text extraction failed for {document.id}: {e}")
        extracted_text = ""

    if extracted_text.strip():
        background_tasks.add_task(index_vault_document, document.id, extracted_text)
    else:
        document.indexing_status = "failed"
        session.add(document)
        session.commit()

    return document.to_dict()


@router.get("/documents")
def list_documents(
    document_type: Optional[str] = None,
    related_case_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    accessible_ids = _accessible_ids_subquery(session, current_user.id)
    conditions = [VaultDocument.user_id == current_user.id]
    if accessible_ids:
        conditions.append(VaultDocument.id.in_(accessible_ids))

    stmt = select(VaultDocument).where(or_(*conditions))
    if document_type:
        stmt = stmt.where(VaultDocument.document_type == document_type)
    if related_case_id:
        stmt = stmt.where(VaultDocument.related_case_id == related_case_id)
    stmt = stmt.order_by(VaultDocument.created_at.desc())

    documents = session.exec(stmt).all()
    return [d.to_dict() for d in documents]


@router.get("/documents/{document_id}")
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    document = _get_accessible_document(session, document_id, current_user)
    storage = get_object_storage()
    return {
        **document.to_dict(),
        "downloadUrl": storage.get_download_url(document.object_key, document.id),
    }


@router.get("/documents/local-download/{document_id}")
def local_download(
    document_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Dev-only: serves files from the local-disk storage fallback. Not used
    when R2 is configured (get_download_url() returns a real presigned URL
    instead of this path). Takes a document_id (not the raw storage key) so
    the same ownership/share check as every other document endpoint applies
    before any bytes are served."""
    storage = get_object_storage()
    if not isinstance(storage, LocalDiskObjectStorage):
        raise MessageHTTPException(status_code=404, detail="Not found")
    document = _get_accessible_document(session, document_id, current_user)
    try:
        data = storage.read_object(document.object_key)
    except (FileNotFoundError, ValueError):
        # ValueError == the key escaped LOCAL_VAULT_DIR (path-traversal attempt).
        raise MessageHTTPException(status_code=404, detail="Not found")
    return Response(content=data, media_type="application/octet-stream")


@router.post("/search")
async def search_documents(
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    accessible_ids = _accessible_ids_subquery(session, current_user.id)
    own_ids = session.exec(
        select(VaultDocument.id).where(VaultDocument.user_id == current_user.id)
    ).all()
    all_ids = set(own_ids) | set(accessible_ids)
    if not all_ids:
        return []

    embeddings_model = await _get_shared_embeddings()
    loop = asyncio.get_event_loop()
    query_vector = await loop.run_in_executor(None, embeddings_model.embed_query, body.query)

    distance_col = VaultDocumentEmbedding.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(VaultDocumentEmbedding, distance_col)
        .where(VaultDocumentEmbedding.vault_document_id.in_(all_ids))
        .order_by(distance_col)
        .limit(10)
    )
    rows = session.exec(stmt).all()

    results = []
    seen_docs = set()
    for embedding, distance in rows:
        if embedding.vault_document_id in seen_docs:
            continue
        seen_docs.add(embedding.vault_document_id)
        document = session.get(VaultDocument, embedding.vault_document_id)
        if document is None:
            continue
        results.append(
            {
                **document.to_dict(),
                "matchedSnippet": embedding.chunk_text[:400],
                "relevance": round(1 - float(distance), 4),
            }
        )
    return results


@router.post("/documents/{document_id}/share")
def share_document(
    document_id: str,
    body: ShareRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    document = session.get(VaultDocument, document_id)
    if document is None or document.user_id != current_user.id:
        raise MessageHTTPException(status_code=404, detail="Document not found")

    target_user = session.get(User, body.sharedWithUserId)
    if target_user is None:
        raise MessageHTTPException(status_code=404, detail="User not found")

    permission = VaultDocumentPermission(
        vault_document_id=document_id,
        shared_with_user_id=body.sharedWithUserId,
        permission=body.permission,
    )
    session.add(permission)
    session.commit()
    session.refresh(permission)

    # Smart Notifications (Phase 4): document_uploaded/shared producer —
    # reuses the same spine Hearing Reminders uses (see
    # app/services/notification_dispatch.py). Best-effort: sharing already
    # succeeded above regardless of whether the notification sends.
    try:
        send_notification(
            session,
            user_id=body.sharedWithUserId,
            type_="document_shared",
            title=f'"{document.title}" was shared with you',
            body=f"{current_user.name} shared a document in your Legal Document Vault.",
            channels=["in_app", "email"],
        )
    except Exception as e:
        print(f"[Vault] share notification failed: {e}")

    return permission.to_dict()


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    document = session.get(VaultDocument, document_id)
    if document is None or document.user_id != current_user.id:
        raise MessageHTTPException(status_code=404, detail="Document not found")

    storage = get_object_storage()
    try:
        storage.delete_object(document.object_key)
    except Exception as e:
        print(f"[Vault] object delete failed for {document.object_key}: {e}")

    session.delete(document)
    session.commit()
    return {"message": "Document deleted"}
