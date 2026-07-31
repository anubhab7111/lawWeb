"""
SQLModel models mirroring the tables created by app/db/schema.sql.

Column names are snake_case (as in the schema); API responses use the
camelCase keys the client expects — see the to_dict() helpers.
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Numeric, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY
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
