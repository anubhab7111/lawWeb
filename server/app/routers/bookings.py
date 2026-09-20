"""
Booking/payment endpoints using the Braintree Python SDK (Sandbox),
ported from the old Express server/src/routes/bookings.ts.
Credentials come strictly from settings (.env) — no hardcoded fallbacks.
"""

import math
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Optional

import braintree
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.config import get_settings
from app.db.engine import get_session
from app.db.models import Booking, BookingStatus, CalendarEvent, Lawyer, User
from app.deps.auth import get_current_user
from app.deps.errors import MessageHTTPException

router = APIRouter(prefix="/api/bookings", tags=["bookings"])


@lru_cache()
def get_gateway() -> braintree.BraintreeGateway:
    settings = get_settings()
    return braintree.BraintreeGateway(
        braintree.Configuration(
            braintree.Environment.Sandbox,
            merchant_id=settings.braintree_merchant_id,
            public_key=settings.braintree_public_key,
            private_key=settings.braintree_private_key,
        )
    )


class CheckoutRequest(BaseModel):
    amount: Optional[str] = None
    paymentMethodNonce: Optional[str] = None
    lawyerId: Optional[str] = None
    userId: Optional[str] = None
    appointmentDate: Optional[str] = None  # YYYY-MM-DD
    appointmentTime: Optional[str] = None  # HH:MM, half-hour slots


@router.get("/client_token")
def client_token():
    """One-time token authorizing the frontend to render the payment UI."""
    try:
        result = get_gateway().client_token.generate({})
        # The client reads this with response.text()
        return PlainTextResponse(result)
    except Exception as e:
        print(f"Braintree Token Error: {e}")
        return PlainTextResponse(
            "Braintree Authentication Failed. Check your API keys.", status_code=500
        )


_SLOT_TIMES = {f"{h:02d}:{m:02d}" for h in range(9, 18) for m in (0, 30)} | {"18:00"}
_IST = timezone(timedelta(hours=5, minutes=30))


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "message": message})


def _parse_slot(
    date_str: Optional[str], time_str: Optional[str]
) -> tuple[Optional[date], Optional[str], Optional[str]]:
    """Returns (date, time, error message)."""
    if not date_str or not time_str:
        return None, None, "Choose an appointment date and time"
    try:
        day = date.fromisoformat(date_str)
    except ValueError:
        return None, None, "Invalid appointment date"
    if day < datetime.now(_IST).date():
        return None, None, "Appointment date must not be in the past"
    if time_str not in _SLOT_TIMES:
        return None, None, "Appointment time must be a half-hour slot between 09:00 and 18:00"
    return day, time_str, None


@router.post("/checkout")
def checkout(
    body: CheckoutRequest,
    idempotency_key: Optional[str] = Header(default=None),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Charge the nonce received from the client and record the booking.

    The booking is always attributed to the authenticated caller — body.userId
    is ignored — so a caller can't create/charge bookings on someone else's
    behalf. The amount is recomputed server-side (see below) rather than trusted.
    A repeated Idempotency-Key returns the original result without charging again.
    """
    if idempotency_key:
        prior = session.exec(
            select(Booking).where(
                Booking.idempotency_key == idempotency_key, Booking.user_id == current_user.id
            )
        ).first()
        if prior is not None:
            if prior.status == BookingStatus.confirmed:
                return {"status": "success", "transactionId": prior.transaction_id}
            return _error(409, "This payment is already being processed")

    # ── Validate BEFORE charging so a bad request never captures money ──
    if not body.amount or not body.paymentMethodNonce or not body.lawyerId:
        return _error(400, "Missing required checkout fields")
    try:
        amount = Decimal(str(body.amount))
    except (InvalidOperation, TypeError):
        return _error(400, "Invalid amount")
    day, slot, slot_error = _parse_slot(body.appointmentDate, body.appointmentTime)
    if slot_error:
        return _error(400, slot_error)
    lawyer = session.get(Lawyer, body.lawyerId)
    if lawyer is None:
        return _error(400, "Lawyer not found")

    # The client charges hourly_rate + a 5% platform fee (client Payment.tsx).
    # Recompute the total here and reject any mismatch so the client can't
    # dictate the price. Mirror JS Math.round (round-half-up) with floor(x+0.5)
    # rather than Python's banker's rounding, or an even .5 fee would diverge.
    fee = math.floor(lawyer.hourly_rate * 0.05 + 0.5)
    expected_total = Decimal(lawyer.hourly_rate + fee)
    if amount != expected_total:
        return _error(400, "Amount does not match the lawyer's rate")

    # Reserve the slot with a pending row first: it blocks a concurrent booking
    # of the same slot and survives a crash between charge and confirmation.
    taken = session.exec(
        select(Booking.id).where(
            Booking.lawyer_id == lawyer.id,
            Booking.appointment_date == day.isoformat(),
            Booking.appointment_time == slot,
            Booking.status.in_([BookingStatus.pending, BookingStatus.confirmed]),
        )
    ).first()
    if taken is not None:
        return _error(409, "That time slot is already booked — please choose another")

    booking = Booking(
        user_id=current_user.id,
        lawyer_id=lawyer.id,
        amount=amount,
        status=BookingStatus.pending,
        transaction_id=f"pending-{uuid.uuid4()}",
        appointment_date=day.isoformat(),
        appointment_time=slot,
        idempotency_key=idempotency_key,
    )
    session.add(booking)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return _error(409, "This payment is already being processed")
    booking_id = booking.id

    def release(status: BookingStatus) -> None:
        session.rollback()
        row = session.get(Booking, booking_id)
        if row is not None:
            row.status = status
            session.add(row)
            session.commit()

    # ── Charge (own error scope: a failure here means no money captured) ──
    try:
        result = get_gateway().transaction.sale(
            {
                "amount": str(amount),
                "payment_method_nonce": body.paymentMethodNonce,
                "options": {"submit_for_settlement": True},
            }
        )
    except Exception as e:
        print(f"Checkout charge error: {e}")
        release(BookingStatus.failed)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error during checkout"},
        )

    if not result.is_success:
        print(f"❌ Braintree Transaction Failed: {result.message}")
        release(BookingStatus.failed)
        return JSONResponse(
            status_code=400, content={"status": "error", "message": result.message}
        )

    # ── Charge succeeded: confirm the booking in a SEPARATE error scope ──
    # A DB failure here must NOT be reported as a failed payment (the card was
    # already charged), or the user will retry and be double-charged. The
    # pending row remains for manual reconciliation.
    transaction_id = result.transaction.id
    try:
        booking = session.get(Booking, booking_id)
        booking.transaction_id = transaction_id
        booking.status = BookingStatus.confirmed
        session.add(booking)
        session.commit()
        print(f"✅ Success: Payment settled for User {current_user.id}")

        # Personal Legal Calendar: best-effort, isolated from the payment
        # result — a calendar-write failure must never turn a successful
        # payment into an error response.
        try:
            start_at = datetime.fromisoformat(
                f"{booking.appointment_date}T{booking.appointment_time}"
            ).replace(tzinfo=_IST)
            session.add(
                CalendarEvent(
                    user_id=booking.user_id,
                    title=f"Consultation with {lawyer.name}",
                    event_type="lawyer_meeting",
                    start_at=start_at,
                    end_at=start_at + timedelta(hours=1),
                    related_booking_id=booking.id,
                )
            )
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[Calendar] failed to auto-create event for booking {booking_id}: {e}")

        return {"status": "success", "transactionId": transaction_id}
    except Exception as e:
        session.rollback()
        # RECONCILIATION: money captured but booking not confirmed. Log loudly
        # so this can be reconciled manually against Braintree settlements.
        print(
            f"⚠️ RECONCILIATION NEEDED: charged transaction {transaction_id} "
            f"for user {current_user.id} / lawyer {body.lawyerId} (booking {booking_id}) "
            f"but confirmation failed: {e}"
        )
        return {
            "status": "success",
            "transactionId": transaction_id,
            "warning": "booking_record_failed",
        }


@router.get("/user-bookings/{user_id}")
def user_bookings(
    user_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Confirmed appointments for a user, newest first. Scoped to the caller —
    the path id must match the authenticated user (no cross-user reads)."""
    if user_id != current_user.id:
        raise MessageHTTPException(status_code=404, detail="Not found")
    try:
        bookings = session.exec(
            select(Booking)
            .where(Booking.user_id == user_id)
            .where(Booking.status == BookingStatus.confirmed)
            .order_by(Booking.created_at.desc())
        ).all()
        return [booking.to_dict() for booking in bookings]
    except Exception as e:
        print(f"Fetch Bookings Error: {e}")
        return JSONResponse(status_code=500, content={"message": "Fetch failed"})
