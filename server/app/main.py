"""
FastAPI backend for the legal chatbot.
Provides REST API endpoints for chat, document upload, and search.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import asyncio
import json

from app.config import get_settings
from app.chatbot import get_chatbot
from app.tools.document_extractor import get_document_extractor
from app.tools.crime_reporter import CRIME_TYPES
from app.tools.lawyer_finder import get_lawyer_finder


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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Legal Chatbot API",
    description="AI-powered legal assistant for document analysis, crime reporting guidance, and lawyer search.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Serve the frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Processes user messages and returns AI responses.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())

        result = await chatbot.chat(message=request.message, session_id=session_id)

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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    Streams LLM response tokens as they are generated.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            chatbot = get_chatbot()
            async for event in chatbot.stream_chat(
                message=request.message,
                session_id=session_id,
            ):
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


@app.post("/chat/upload", response_model=ChatResponse)
async def chat_with_document(
    file: UploadFile = File(
        ..., description="Document file (PDF, DOCX, TXT, JPG, PNG)"
    ),
    message: str = Form(
        default="Please analyze this document", description="User message"
    ),
    session_id: Optional[str] = Form(default=None, description="Session ID"),
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

        result = await chatbot.chat(
            message=message,
            session_id=session_id,
            document_content=document_text,
            document_type=doc_type,  # Pass document type for pipeline
        )

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


@app.post("/analyze-document", response_model=ChatResponse)
async def analyze_document_text(request: DocumentAnalysisRequest):
    """
    Analyze document text directly without file upload.
    Useful when document text is already extracted.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())

        result = await chatbot.chat(
            message="Please analyze this document thoroughly.",
            session_id=session_id,
            document_content=request.document_text,
        )

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


class DocumentValidationRequest(BaseModel):
    """Request for statutory compliance validation of a legal document."""

    document_text: str = Field(
        ..., description="Document text to validate", min_length=10
    )
    session_id: Optional[str] = Field(None, description="Session ID")


@app.post("/validate-document", response_model=ChatResponse)
async def validate_document_text(request: DocumentValidationRequest):
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

        result = await chatbot.chat(
            message="Please validate this document for statutory compliance.",
            session_id=session_id,
            document_content=request.document_text,
            document_type="text",
        )

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


@app.post("/validate-document/upload", response_model=ChatResponse)
async def validate_document_upload(
    file: UploadFile = File(
        ..., description="Document file (PDF, DOCX, TXT, JPG, PNG)"
    ),
    message: str = Form(
        default="Please validate this document for statutory compliance",
        description="User message",
    ),
    session_id: Optional[str] = Form(default=None, description="Session ID"),
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

        result = await chatbot.chat(
            message=validation_message,
            session_id=session_id,
            document_content=document_text,
            document_type=doc_type,
        )

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


@app.post("/crime-report", response_model=ChatResponse)
async def get_crime_report_guidance(request: CrimeReportRequest):
    """
    Get guidance for reporting a crime.
    Returns structured steps and resources.
    """
    try:
        chatbot = get_chatbot()
        session_id = request.session_id or str(uuid.uuid4())

        result = await chatbot.chat(
            message=f"I need help reporting a crime: {request.description}",
            session_id=session_id,
        )

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


@app.post("/find-lawyer")
async def find_lawyers(request: LawyerSearchRequest):
    """
    Search for lawyers based on criteria.
    Returns matching lawyers without LLM processing for speed.
    """
    try:
        finder = get_lawyer_finder()

        # If specialization not provided, try to detect from query
        specialization = request.specialization
        if not specialization:
            specialization = finder.detect_specialization(request.query)

        lawyers = finder.search_lawyers(
            specialization=specialization, location=request.location, limit=10
        )

        return {
            "lawyers": [
                {
                    "name": l.name,
                    "specialization": l.specialization,
                    "location": l.location,
                    "contact": l.contact,
                    "email": l.email,
                    "rating": l.rating,
                    "experience_years": l.experience_years,
                    "languages": l.languages,
                    "consultation_fee": l.consultation_fee,
                    "bar_number": l.bar_number,
                }
                for l in lawyers
            ],
            "count": len(lawyers),
            "detected_specialization": specialization,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lawyer search error: {str(e)}")


@app.get("/specializations")
async def get_specializations():
    """Get list of available legal specializations."""
    finder = get_lawyer_finder()
    return {"specializations": finder.get_specializations()}


@app.get("/crime-types")
async def get_crime_types():
    """Get list of recognized crime types."""
    return {"crime_types": CRIME_TYPES}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session's history."""
    try:
        chatbot = get_chatbot()
        chatbot.clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get the message history for a session."""
    try:
        chatbot = get_chatbot()
        history = chatbot.get_session_history(session_id)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": (
                str(exc) if get_settings().llm_temperature > 0 else "An error occurred"
            ),
        },
    )


# ============================================================================
# Main Entry Point
# ============================================================================


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
