"""
SQLModel models mirroring the tables created by app/db/schema.sql.

Column names are snake_case (as in the schema); API responses use the
camelCase keys the client expects — see the to_dict() helpers.
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Numeric, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Field, SQLModel


class BookingStatus(str, enum.Enum):
    pending = "pending"
    confirmed = "confirmed"
    failed = "failed"


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    email: str = Field(unique=True)
    password: str
    # "client" | "lawyer" | "admin". Not in to_dict() for now — the client
    # doesn't read it yet. get_current_user() (app/deps/auth.py) already
    # loads the full User row, so lawyer-scoped features (Vault sharing,
    # notifications) can check `.role` there without a second query.
    role: str = Field(default="client")
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )


class Lawyer(SQLModel, table=True):
    __tablename__ = "lawyers"

    id: str = Field(primary_key=True)
    name: str
    specialty: str
    experience: int
    rating: float
    hourly_rate: int
    location: str
    bio: str
    cases: int
    success_rate: int
    education: str
    languages: List[str] = Field(sa_column=Column(ARRAY(Text)))
    availability: str
    # Nullable link to a real login, so a directory listing can optionally be
    # claimed by a lawyer who signs up (needed for Vault sharing / "your
    # lawyer uploaded a document" notifications). Directory entries with no
    # claimed account keep user_id NULL.
    user_id: Optional[str] = Field(default=None, foreign_key="users.id")
    # Embedding of "{specialty}. {bio}" via BAAI/bge-large-en-v1.5 (1024-dim),
    # used for semantic matching in recommend_lawyers(). Internal only — never
    # serialized in to_dict().
    bio_embedding: Optional[List[float]] = Field(
        default=None, sa_column=Column(Vector(1024), nullable=True)
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "specialty": self.specialty,
            "experience": self.experience,
            "rating": self.rating,
            "hourlyRate": self.hourly_rate,
            "location": self.location,
            "bio": self.bio,
            "cases": self.cases,
            "successRate": self.success_rate,
            "education": self.education,
            "languages": self.languages,
            "availability": self.availability,
        }


class Booking(SQLModel, table=True):
    __tablename__ = "bookings"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    lawyer_id: str = Field(foreign_key="lawyers.id")
    amount: Decimal = Field(sa_column=Column(Numeric(10, 2)))
    transaction_id: str
    status: BookingStatus = Field(
        default=BookingStatus.pending,
        sa_column=Column(SAEnum(BookingStatus, name="BookingStatus")),
    )
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "lawyerId": self.lawyer_id,
            "amount": float(self.amount),
            "transactionId": self.transaction_id,
            "status": self.status.value if self.status else None,
            "appointmentDate": self.appointment_date,
            "appointmentTime": self.appointment_time,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Similar Case Search (Phase 1)
# ============================================================================


class SimilarCaseSearch(SQLModel, table=True):
    __tablename__ = "similar_case_searches"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    uploaded_doc_summary: str
    firac: Dict[str, Any] = Field(sa_column=Column(JSONB))
    results: Dict[str, Any] = Field(sa_column=Column(JSONB))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "uploadedDocSummary": self.uploaded_doc_summary,
            "firac": self.firac,
            "results": self.results,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# My Cases (Phase 2)
# ============================================================================


class SavedCase(SQLModel, table=True):
    __tablename__ = "saved_cases"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    cnr: Optional[str] = None
    court: Optional[str] = None
    case_number: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    status: Optional[str] = None
    last_synced_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "cnr": self.cnr,
            "court": self.court,
            "caseNumber": self.case_number,
            "year": self.year,
            "title": self.title,
            "status": self.status,
            "lastSyncedAt": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class CaseEvent(SQLModel, table=True):
    __tablename__ = "case_events"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    saved_case_id: str = Field(foreign_key="saved_cases.id")
    event_type: str  # "hearing" | "order" | "filing" | ...
    event_date: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    title: Optional[str] = None
    detail: Optional[str] = None
    source_url: Optional[str] = None
    raw_payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "savedCaseId": self.saved_case_id,
            "eventType": self.event_type,
            "eventDate": self.event_date.isoformat() if self.event_date else None,
            "title": self.title,
            "detail": self.detail,
            "sourceUrl": self.source_url,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class CaseNote(SQLModel, table=True):
    __tablename__ = "case_notes"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    saved_case_id: str = Field(foreign_key="saved_cases.id")
    user_id: str = Field(foreign_key="users.id")
    note_text: str
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "savedCaseId": self.saved_case_id,
            "userId": self.user_id,
            "noteText": self.note_text,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class CaseAiSummary(SQLModel, table=True):
    __tablename__ = "case_ai_summaries"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    saved_case_id: str = Field(foreign_key="saved_cases.id")
    summary_text: str
    source_event_id: Optional[str] = Field(default=None, foreign_key="case_events.id")
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "savedCaseId": self.saved_case_id,
            "summaryText": self.summary_text,
            "sourceEventId": self.source_event_id,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Notifications (Phase 2, generalized by Phase 4)
# ============================================================================


class NotificationPreference(SQLModel, table=True):
    __tablename__ = "notification_preferences"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id", unique=True)
    email_enabled: bool = Field(default=True)
    push_enabled: bool = Field(default=False)
    sms_enabled: bool = Field(default=False)
    fcm_token: Optional[str] = None
    phone_number: Optional[str] = None
    # Per-notification-type overrides, e.g. {"new_order": {"email": false}} —
    # avoids a second table while still allowing "email me for hearings but
    # not for document uploads" once producers multiply in Phase 4.
    type_overrides: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "emailEnabled": self.email_enabled,
            "pushEnabled": self.push_enabled,
            "smsEnabled": self.sms_enabled,
            "hasFcmToken": bool(self.fcm_token),
            "phoneNumber": self.phone_number,
            "typeOverrides": self.type_overrides,
        }


class Notification(SQLModel, table=True):
    __tablename__ = "notifications"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    type: str
    title: str
    body: str
    related_case_id: Optional[str] = Field(default=None, foreign_key="saved_cases.id")
    channel: str  # "email" | "push" | "sms" | "in_app"
    status: str = Field(default="pending")  # "pending" | "sent" | "failed"
    scheduled_for: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    sent_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    read_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "type": self.type,
            "title": self.title,
            "body": self.body,
            "relatedCaseId": self.related_case_id,
            "channel": self.channel,
            "status": self.status,
            "scheduledFor": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "sentAt": self.sent_at.isoformat() if self.sent_at else None,
            "readAt": self.read_at.isoformat() if self.read_at else None,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


class CauseListCache(SQLModel, table=True):
    __tablename__ = "cause_list_cache"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    court: str
    list_date: str  # ISO date (YYYY-MM-DD); stored as text like bookings' appointment_date
    fetched_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    entries: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "court": self.court,
            "listDate": self.list_date,
            "fetchedAt": self.fetched_at.isoformat() if self.fetched_at else None,
            "entries": self.entries,
        }


# ============================================================================
# Legal Document Vault (Phase 3)
# ============================================================================


class VaultDocument(SQLModel, table=True):
    __tablename__ = "vault_documents"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    title: str
    document_type: str  # "fir" | "order" | "agreement" | "notice" | "evidence" | "judgment" | ...
    object_key: str  # storage key (R2 or local-disk fallback — see app/services/object_storage.py)
    file_size_bytes: int
    mime_type: str
    related_case_id: Optional[str] = Field(default=None, foreign_key="saved_cases.id")
    extracted_text: Optional[str] = None
    indexing_status: str = Field(default="pending")  # "pending" | "indexing" | "ready" | "failed"
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "title": self.title,
            "documentType": self.document_type,
            "fileSizeBytes": self.file_size_bytes,
            "mimeType": self.mime_type,
            "relatedCaseId": self.related_case_id,
            "indexingStatus": self.indexing_status,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class VaultDocumentEmbedding(SQLModel, table=True):
    __tablename__ = "vault_document_embeddings"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    vault_document_id: str = Field(foreign_key="vault_documents.id")
    chunk_index: int
    chunk_text: str
    embedding: List[float] = Field(sa_column=Column(Vector(1024)))


class VaultDocumentPermission(SQLModel, table=True):
    __tablename__ = "vault_document_permissions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    vault_document_id: str = Field(foreign_key="vault_documents.id")
    shared_with_user_id: str = Field(foreign_key="users.id")
    permission: str = Field(default="view")  # "view" | "edit"
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vaultDocumentId": self.vault_document_id,
            "sharedWithUserId": self.shared_with_user_id,
            "permission": self.permission,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Personal Legal Calendar (Phase 4)
# ============================================================================


class CalendarEvent(SQLModel, table=True):
    __tablename__ = "calendar_events"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    title: str
    event_type: str  # "hearing" | "deadline" | "lawyer_meeting" | "custom"
    start_at: datetime = Field(sa_column=Column(DateTime(timezone=True)))
    end_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    related_case_id: Optional[str] = Field(default=None, foreign_key="saved_cases.id")
    related_booking_id: Optional[str] = Field(default=None, foreign_key="bookings.id")
    google_calendar_event_id: Optional[str] = None
    outlook_event_id: Optional[str] = None
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "title": self.title,
            "eventType": self.event_type,
            "startAt": self.start_at.isoformat() if self.start_at else None,
            "endAt": self.end_at.isoformat() if self.end_at else None,
            "relatedCaseId": self.related_case_id,
            "relatedBookingId": self.related_booking_id,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
