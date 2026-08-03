"""
Chatbot endpoints, moved from app/main.py and mounted under /api/chat so the
client keeps the exact paths it used through the old Express proxy.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.chatbot import get_chatbot
from app.config import get_settings
from app.db.engine import get_session
from app.db.models import ChatMessage, ChatSession, MessageRole, User
from app.deps.auth import get_current_user, get_current_user_optional
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


async def _seed_from_db_if_needed(
    session: Session, chatbot, user: Optional[User], session_id: str
) -> None:
    """Load prior DB history into the in-memory cache for an authenticated
    user whose session_id isn't already live in this process (e.g. after a
    server restart)."""
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
) -> None:
    """No-op for guests. For authenticated users: create the chat_sessions
    row if absent, then append both turns to chat_messages. If session_id
    already belongs to a different user (collision/reuse across accounts),
    silently skip persistence — the chat call itself must still succeed."""
    if user is None or not assistant_message:
        return

    chat_session = session.get(ChatSession, session_id)
    if chat_session is not None and chat_session.user_id != user.id:
        return

    if chat_session is None:
        chat_session = ChatSession(id=session_id, user_id=user.id, title=user_message[:80])
        session.add(chat_session)
        # Without a declared relationship() between ChatSession and
        # ChatMessage, SQLAlchemy's unit-of-work doesn't order inserts by the
        # plain FK column alone — flush the parent row explicitly so the
        # chat_messages insert below doesn't violate the FK constraint.
        session.flush()

    session.add(ChatMessage(session_id=session_id, role=MessageRole.user, content=user_message))
    session.add(
        ChatMessage(session_id=session_id, role=MessageRole.assistant, content=assistant_message)
    )
    chat_session.updated_at = datetime.now(timezone.utc)
    session.add(chat_session)
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

        await _seed_from_db_if_needed(session, chatbot, user, session_id)
        result = await chatbot.chat(message=request.message, session_id=session_id)
        await _persist_turn(session, user, session_id, request.message, result.get("response", ""))

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
    session: Session = Depends(get_session),
):
    """
    Streaming chat endpoint using Server-Sent Events.
    Streams LLM response tokens as they are generated.
    """
    session_id = request.session_id or str(uuid.uuid4())
    chatbot = get_chatbot()
    await _seed_from_db_if_needed(session, chatbot, user, session_id)

    async def event_generator():
        try:
            async for event in chatbot.stream_chat(
                message=request.message,
                session_id=session_id,
            ):
                if event.get("type") == "done":
                    await _persist_turn(
                        session, user, session_id, request.message, event.get("response", "")
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
async def stop_stream(request: StopStreamRequest):
    """
    Stop button: cancels an in-flight /stream generation for this session.
    Closing the client's fetch/EventSource alone would not do this — the
    handler runs as a detached asyncio task so it keeps generating (and
    burning LLM compute) even after the HTTP response is abandoned.
    """
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
        file_bytes = await file.read()

        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.max_document_size_mb}MB",
            )

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
        file_bytes = await file.read()

        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.max_document_size_mb}MB",
            )

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
        chatbot = get_chatbot()
        chatbot.clear_session(session_id)

        if user is not None:
            chat_session = session.get(ChatSession, session_id)
            if chat_session is not None and chat_session.user_id == user.id:
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
    for a session the current user owns; otherwise falls back to the
    in-memory (20-message-capped) history, same as before this feature."""
    try:
        if user is not None:
            chat_session = session.get(ChatSession, session_id)
            if chat_session is not None and chat_session.user_id == user.id:
                rows = session.exec(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at)
                ).all()
                messages = [{"role": r.role.value, "content": r.content} for r in rows]
                return {"session_id": session_id, "messages": messages, "count": len(messages)}

        chatbot = get_chatbot()
        history = chatbot.get_session_history(session_id)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")
