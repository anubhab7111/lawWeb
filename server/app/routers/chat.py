"""
Chatbot endpoints, moved from app/main.py and mounted under /api/chat so the
client keeps the exact paths it used through the old Express proxy.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.chatbot import get_chatbot
from app.config import get_settings
from app.db.engine import get_engine, get_session
from app.db.models import ChatMessage, ChatSession, MessageRole, User
from app.deps.auth import get_current_user, get_current_user_optional
from app.deps.uploads import read_upload_within_limit
from app.tools.crime_reporter import CRIME_TYPES
from app.tools.document_extractor import get_document_extractor
from app.tools.lawyer_recommender import (
    LEGAL_SPECIALIZATIONS,
    recommend_lawyers as recommend_lawyers_core,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ============================================================================
# Pydantic Models
# ============================================================================


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ..., description="User's message", min_length=1, max_length=5000
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation context"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID")
    intent: Optional[str] = Field(None, description="Detected intent")
    document_info: Optional[Dict[str, Any]] = Field(
        None, description="Document analysis info if applicable"
    )
    document_validation: Optional[Dict[str, Any]] = Field(
        None, description="Document validation info from 3-layer pipeline"
    )
    crime_report: Optional[Dict[str, Any]] = Field(
        None, description="Crime report info if applicable"
    )
    lawyers_found: Optional[List[Dict[str, Any]]] = Field(
        None, description="Found lawyers if applicable"
    )


class DocumentAnalysisRequest(BaseModel):
    """Request for analyzing document text directly."""

    document_text: str = Field(
        ..., description="Document text to analyze", min_length=10
    )
    session_id: Optional[str] = Field(None, description="Session ID")


class CrimeReportRequest(BaseModel):
    """Request for crime reporting guidance."""

    description: str = Field(
        ..., description="Description of the crime/incident", min_length=10
    )
    session_id: Optional[str] = Field(None, description="Session ID")


class LawyerSearchRequest(BaseModel):
    """Request for lawyer search."""

    query: str = Field(
        ..., description="Search query for finding lawyers", min_length=2
    )
    location: Optional[str] = Field(None, description="Preferred location")
    specialization: Optional[str] = Field(
        None, description="Legal specialization needed"
    )


class DocumentValidationRequest(BaseModel):
    """Request for statutory compliance validation of a legal document."""

    document_text: str = Field(
        ..., description="Document text to validate", min_length=10
    )
    session_id: Optional[str] = Field(None, description="Session ID")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


# ============================================================================
# DB-backed history (logged-in users only; guests stay in-memory-only, see
# app.chatbot.LegalChatbot._sessions)
# ============================================================================


async def _resolve_session_id(session: Session, user: Optional[User], session_id: str) -> str:
    """If session_id belongs to a different account, mint a fresh one instead
    of reusing it. This must run before the chatbot is ever invoked: the
    in-memory LangGraph cache (LegalChatbot._sessions) is a single
    process-wide dict keyed only by session_id, with no per-user isolation —
    so silently skipping the DB write on a collision (as _persist_turn does)
    isn't enough on its own, it would still hand one account's live
    conversation context to whoever guessed/reused the id."""
    if user is None:
        return session_id
    chat_session = session.get(ChatSession, session_id)
    if chat_session is not None and chat_session.user_id != user.id:
        return str(uuid.uuid4())
    return session_id


async def _seed_from_db_if_needed(
    session: Session, chatbot, user: Optional[User], session_id: str
) -> None:
    """Load prior DB history into the in-memory cache for an authenticated
    user whose session_id isn't already live in this process (e.g. after a
    server restart). Assumes session_id has already been through
    _resolve_session_id, so any DB row found here is guaranteed owned by
    `user`."""
    if user is None or chatbot.has_session(session_id):
        return
    chat_session = session.get(ChatSession, session_id)
    if chat_session is None or chat_session.user_id != user.id:
        return
    rows = session.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    ).all()
    chatbot.seed_session(
        session_id, [{"role": r.role.value, "content": r.content} for r in rows]
    )


async def _persist_turn(
    session: Session,
    user: Optional[User],
    session_id: str,
    user_message: str,
    assistant_message: str,
    language: str = "en",
    user_message_display: Optional[str] = None,
    assistant_message_display: Optional[str] = None,
) -> None:
    """No-op for guests. For authenticated users: create the chat_sessions
    row if absent, then append both turns to chat_messages. Assumes
    session_id has already been through _resolve_session_id; the ownership
    check below is a defensive backstop, not the primary guard.

    `user_message`/`assistant_message` are the canonical English text stored in
    `content` (memory is language-independent). For a non-English turn,
    `*_display` carry the original-language text the user typed/saw and
    `language` its ISO code, stored in content_display/language so history
    re-renders in the user's language. English turns pass None displays."""
    if user is None or not assistant_message:
        return

    chat_session = session.get(ChatSession, session_id)
    if chat_session is not None and chat_session.user_id != user.id:
        return

    # Title from what the user actually typed (original language), not the
    # English translation, so the sidebar shows a recognisable entry.
    title_source = user_message_display or user_message

    if chat_session is None:
        chat_session = ChatSession(id=session_id, user_id=user.id, title=title_source[:80])
        session.add(chat_session)
        try:
            # Without a declared relationship() between ChatSession and
            # ChatMessage, SQLAlchemy's unit-of-work doesn't order inserts by
            # the plain FK column alone — flush the parent row explicitly so
            # the chat_messages insert below doesn't violate the FK
            # constraint.
            session.flush()
        except IntegrityError:
            # Two concurrent first-turns for the same brand-new session_id
            # (double submit / client retry) both pass the `chat_session is
            # None` check above; the loser's insert hits the primary-key
            # conflict here. Recover by rolling back and picking up the
            # winner's row instead of surfacing a 500 for an otherwise
            # successful chat response.
            session.rollback()
            chat_session = session.get(ChatSession, session_id)
            if chat_session is None or chat_session.user_id != user.id:
                return

    # user/assistant messages are inserted in one transaction, and Postgres's
    # now()/CURRENT_TIMESTAMP returns the transaction start time for every
    # statement in it — both rows would otherwise get an identical
    # created_at, leaving their relative order (used when reseeding history)
    # unspecified. Set explicit, strictly-increasing timestamps instead.
    user_turn_at = datetime.now(timezone.utc)
    session.add(
        ChatMessage(
            session_id=session_id,
            role=MessageRole.user,
            content=user_message,
            language=language,
            content_display=user_message_display,
            created_at=user_turn_at,
        )
    )
    session.add(
        ChatMessage(
            session_id=session_id,
            role=MessageRole.assistant,
            content=assistant_message,
            language=language,
            content_display=assistant_message_display,
            created_at=user_turn_at + timedelta(microseconds=1),
        )
    )
    chat_session.updated_at = datetime.now(timezone.utc)
    session.commit()


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Main chat endpoint.
    Processes user messages and returns AI responses.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())
        session_id = await _resolve_session_id(session, user, session_id)

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(message=request.message, session_id=session_id)
        language = result.get("language", "en")
        is_translated = language != "en"
        await _persist_turn(
            session,
            user,
            session_id,
            # Canonical English for memory; original-language text for display.
            user_message=result.get("query_en") or request.message,
            assistant_message=result.get("response_en") or result.get("response", ""),
            language=language,
            user_message_display=request.message if is_translated else None,
            assistant_message_display=result.get("response") if is_translated else None,
        )

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent=result.get("intent"),
            document_info=result.get("document_info"),
            document_validation=result.get("document_validation"),
            crime_report=result.get("crime_report"),
            lawyers_found=result.get("lawyers_found"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Streaming chat endpoint using Server-Sent Events.
    Streams LLM response tokens as they are generated.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            chatbot = get_chatbot()
            # Short-lived DB sessions instead of a Depends(get_session) held
            # for the whole SSE response: an LLM stream can run for minutes,
            # and holding a pooled connection open that long risks starving
            # every other endpoint's connection pool for the duration.
            with Session(get_engine()) as db_session:
                resolved_session_id = await _resolve_session_id(db_session, user, session_id)
                await _seed_from_db_if_needed(db_session, chatbot, user, resolved_session_id)

            async for event in chatbot.stream_chat(
                message=request.message,
                session_id=resolved_session_id,
            ):
                # "stopped" (Stop button cancelled generation mid-stream) also
                # carries a "response" — the partial text already shown to
                # the user — and must be persisted the same as "done", or a
                # stopped turn would silently never reach the DB.
                if event.get("type") in ("done", "stopped"):
                    language = event.get("language", "en")
                    is_translated = language != "en"
                    with Session(get_engine()) as db_session:
                        await _persist_turn(
                            db_session,
                            user,
                            resolved_session_id,
                            user_message=event.get("query_en") or request.message,
                            assistant_message=event.get("response_en")
                            or event.get("response", ""),
                            language=language,
                            user_message_display=request.message if is_translated else None,
                            assistant_message_display=(
                                event.get("response") if is_translated else None
                            ),
                        )
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class StopStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session whose in-flight stream to cancel")


@router.post("/stream/stop")
async def stop_stream(
    request: StopStreamRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Stop button: cancels an in-flight /stream generation for this session.
    Closing the client's fetch/EventSource alone would not do this — the
    handler runs as a detached asyncio task so it keeps generating (and
    burning LLM compute) even after the HTTP response is abandoned.
    """
    chat_session = session.get(ChatSession, request.session_id)
    if chat_session is not None and (user is None or chat_session.user_id != user.id):
        # Same ownership guard as clear_session: don't let an unrelated
        # caller who knows/guesses this session_id cancel another
        # account's in-flight generation.
        return {"stopped": False}

    stopped = get_chatbot().stop_stream(request.session_id)
    return {"stopped": stopped}


@router.post("/upload", response_model=ChatResponse)
async def chat_with_document(
    file: UploadFile = File(
        ..., description="Document file (PDF, DOCX, TXT, JPG, PNG)"
    ),
    message: str = Form(
        default="Please analyze this document", description="User message"
    ),
    session_id: Optional[str] = Form(default=None, description="Session ID"),
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Chat endpoint with document/image upload.
    Extracts text from uploaded documents and images (using OCR) and analyzes them.
    """
    # Validate file size
    settings = get_settings()
    max_size = settings.max_document_size_mb * 1024 * 1024  # Convert to bytes

    # Read file content
    try:
        file_bytes = await read_upload_within_limit(file, max_size)

        # Extract text from document or image
        extractor = get_document_extractor()
        document_text, doc_type = await extractor.extract_text(
            file_bytes, file.filename or "document.txt"
        )

        if not document_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from the file. Please ensure it contains readable text or is a clear image.",
            )

        # Process with chatbot - pass document_type for enhanced analysis
        chatbot = get_chatbot()
        session_id = session_id or str(uuid.uuid4())
        session_id = await _resolve_session_id(session, user, session_id)

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(
            message=message,
            session_id=session_id,
            document_content=document_text,
            document_type=doc_type,  # Pass document type for pipeline
        )
        await _persist_turn(session, user, session_id, message, result.get("response", ""))

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent=result.get("intent", "document_analysis"),
            document_info=result.get("document_info"),
            document_validation=result.get("document_validation"),
            crime_report=None,
            lawyers_found=None,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Document processing error: {str(e)}"
        )


@router.post("/analyze-document", response_model=ChatResponse)
async def analyze_document_text(
    request: DocumentAnalysisRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Analyze document text directly without file upload.
    Useful when document text is already extracted.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())
        session_id = await _resolve_session_id(session, user, session_id)
        analyze_message = "Please analyze this document thoroughly."

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(
            message=analyze_message,
            session_id=session_id,
            document_content=request.document_text,
        )
        await _persist_turn(session, user, session_id, analyze_message, result.get("response", ""))

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent="document_analysis",
            document_info=result.get("document_info"),
            document_validation=result.get("document_validation"),
            crime_report=None,
            lawyers_found=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/validate-document", response_model=ChatResponse)
async def validate_document_text(
    request: DocumentValidationRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Validate a legal document for statutory compliance using the 3-layer pipeline.

    Layer 1: Document Classification (deterministic)
    Layer 2: Statutory Checklist Validation (rule-based)
    Layer 3: Legal Defect Analysis (LLM-based)

    Returns comprehensive compliance report with Act/Section references.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())
        session_id = await _resolve_session_id(session, user, session_id)
        validate_message = "Please validate this document for statutory compliance."

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(
            message=validate_message,
            session_id=session_id,
            document_content=request.document_text,
            document_type="text",
        )
        await _persist_turn(session, user, session_id, validate_message, result.get("response", ""))

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent="document_analysis",
            document_info=None,
            document_validation=result.get("document_validation"),
            crime_report=None,
            lawyers_found=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/validate-document/upload", response_model=ChatResponse)
async def validate_document_upload(
    file: UploadFile = File(
        ..., description="Document file (PDF, DOCX, TXT, JPG, PNG)"
    ),
    message: str = Form(
        default="Please validate this document for statutory compliance",
        description="User message",
    ),
    session_id: Optional[str] = Form(default=None, description="Session ID"),
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Upload a document for statutory compliance validation.
    Extracts text and runs the 3-layer validation pipeline.
    """
    settings = get_settings()
    max_size = settings.max_document_size_mb * 1024 * 1024

    try:
        file_bytes = await read_upload_within_limit(file, max_size)

        extractor = get_document_extractor()
        document_text, doc_type = await extractor.extract_text(
            file_bytes, file.filename or "document.txt"
        )

        if not document_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from the file.",
            )

        # Force validation intent by including keyword in message
        validation_message = (
            message
            if "validate" in message.lower()
            else f"Please validate this document: {message}"
        )

        chatbot = get_chatbot()
        session_id = session_id or str(uuid.uuid4())
        session_id = await _resolve_session_id(session, user, session_id)

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(
            message=validation_message,
            session_id=session_id,
            document_content=document_text,
            document_type=doc_type,
        )
        await _persist_turn(session, user, session_id, validation_message, result.get("response", ""))

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent="document_analysis",
            document_info=None,
            document_validation=result.get("document_validation"),
            crime_report=None,
            lawyers_found=None,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Document validation error: {str(e)}"
        )


@router.post("/crime-report", response_model=ChatResponse)
async def get_crime_report_guidance(
    request: CrimeReportRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Get guidance for reporting a crime.
    Returns structured steps and resources.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())
        session_id = await _resolve_session_id(session, user, session_id)
        crime_message = f"I need help reporting a crime: {request.description}"

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(
            message=crime_message,
            session_id=session_id,
        )
        await _persist_turn(session, user, session_id, crime_message, result.get("response", ""))

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent="crime_report",
            document_info=None,
            document_validation=None,
            crime_report=result.get("crime_report"),
            lawyers_found=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crime report error: {str(e)}")


@router.post("/find-lawyer")
async def find_lawyers(
    request: LawyerSearchRequest, session: Session = Depends(get_session)
):
    """
    Search for lawyers based on criteria (semantic match on the query text,
    fused with rating/success_rate).
    """
    try:
        lawyers = await recommend_lawyers_core(
            session,
            problem_description=request.query,
            specialty=request.specialization,
            location=request.location,
            limit=10,
        )

        return {
            "lawyers": [lawyer.to_dict() for lawyer in lawyers],
            "count": len(lawyers),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lawyer search error: {str(e)}")


@router.get("/specializations")
async def get_specializations():
    """Get list of available legal specializations."""
    return {"specializations": LEGAL_SPECIALIZATIONS}


@router.get("/crime-types")
async def get_crime_types():
    """Get list of recognized crime types."""
    return {"crime_types": CRIME_TYPES}


@router.get("/sessions")
async def list_sessions(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """List the current user's persisted chat sessions, most recent first."""
    try:
        rows = session.exec(
            select(ChatSession)
            .where(ChatSession.user_id == user.id)
            .order_by(ChatSession.updated_at.desc())
        ).all()
        return {"sessions": [s.to_dict() for s in rows], "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Clear a chat session's history (in-memory always; DB rows too if the
    session is owned by the requesting user)."""
    try:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is not None and (user is None or chat_session.user_id != user.id):
            # Tied to someone else's account (or the caller isn't
            # authenticated at all) — don't let an unrelated caller who
            # knows/guesses this session_id wipe another account's live
            # in-memory conversation. Report success anyway (same response
            # shape either way) since from the caller's point of view "this
            # session_id has no state I can see" is indistinguishable from
            # "cleared".
            return {"message": f"Session {session_id} cleared"}

        chatbot = get_chatbot()
        chatbot.clear_session(session_id)

        if chat_session is not None:
            session.delete(chat_session)  # cascades to chat_messages
            session.commit()

        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Get the message history for a session. Returns the full DB transcript
    for a session the current user owns; falls back to the in-memory
    (20-message-capped) history only for a session_id with no DB row at all
    (i.e. genuinely never tied to any account) — never for a session_id that
    belongs to a different account, since _seed_from_db_if_needed may have
    already loaded that account's real transcript into the shared in-memory
    cache."""
    try:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is not None:
            if user is None or chat_session.user_id != user.id:
                return {"session_id": session_id, "messages": [], "count": 0}
            rows = session.exec(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            ).all()
            # Render the user's original-language text (content_display) when
            # present; English turns / legacy rows fall back to content.
            messages = [
                {
                    "role": r.role.value,
                    "content": r.content_display or r.content,
                    "language": r.language,
                }
                for r in rows
            ]
            return {"session_id": session_id, "messages": messages, "count": len(messages)}

        chatbot = get_chatbot()
        history = chatbot.get_session_history(session_id)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")
