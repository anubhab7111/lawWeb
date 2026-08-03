"""
Booking/payment endpoints using the Braintree Python SDK (Sandbox),
ported from the old Express server/src/routes/bookings.ts.
Credentials come strictly from settings (.env) — no hardcoded fallbacks.
"""

import math
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Optional

import braintree
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
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


@router.post("/checkout")
def checkout(
    body: CheckoutRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Charge the nonce received from the client and record the booking.

    The booking is always attributed to the authenticated caller — body.userId
    is ignored — so a caller can't create/charge bookings on someone else's
    behalf. The amount is recomputed server-side (see below) rather than trusted.
    """
    # ── Validate BEFORE charging so a bad request never captures money ──
    if not body.amount or not body.paymentMethodNonce or not body.lawyerId:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Missing required checkout fields"},
        )
    try:
        amount = Decimal(str(body.amount))
    except (InvalidOperation, TypeError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid amount"},
        )
    # Catch the most common cause of a post-charge FK failure before charging.
    lawyer = session.get(Lawyer, body.lawyerId)
    if lawyer is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Lawyer not found"},
        )

    # The client charges hourly_rate + a 5% platform fee (client Payment.tsx).
    # Recompute the total here and reject any mismatch so the client can't
    # dictate the price. Mirror JS Math.round (round-half-up) with floor(x+0.5)
    # rather than Python's banker's rounding, or an even .5 fee would diverge.
    fee = math.floor(lawyer.hourly_rate * 0.05 + 0.5)
    expected_total = Decimal(lawyer.hourly_rate + fee)
    if amount != expected_total:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Amount does not match the lawyer's rate"},
        )

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
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error during checkout"},
        )

    if not result.is_success:
        print(f"❌ Braintree Transaction Failed: {result.message}")
        return JSONResponse(
            status_code=400, content={"status": "error", "message": result.message}
        )

    # ── Charge succeeded: record the booking in a SEPARATE error scope ──
    # A DB failure here must NOT be reported as a failed payment (the card was
    # already charged), or the user will retry and be double-charged. Persist
    # what we can, flag for reconciliation, and still report success.
    transaction_id = result.transaction.id
    try:
        booking = Booking(
            user_id=current_user.id,
            lawyer_id=body.lawyerId,
            amount=amount,
            status=BookingStatus.confirmed,
            transaction_id=transaction_id,
        )
        session.add(booking)
        session.commit()
        print(f"✅ Success: Payment settled for User {current_user.id}")

        # Personal Legal Calendar: auto-add a "lawyer meeting" event for this
        # booking. Best-effort and isolated from the payment/booking result
        # above — a calendar-write failure must never turn a successful
        # payment into an error response. appointment_date/appointment_time
        # aren't collected by this endpoint yet, so this is a no-op until a
        # scheduling step is added to checkout.
        if booking.appointment_date and booking.appointment_time:
            try:
                from datetime import datetime

                start_at = datetime.fromisoformat(
                    f"{booking.appointment_date}T{booking.appointment_time}"
                )
                session.add(
                    CalendarEvent(
                        user_id=booking.user_id,
                        title=f"Consultation with lawyer {booking.lawyer_id}",
                        event_type="lawyer_meeting",
                        start_at=start_at,
                        related_booking_id=booking.id,
                    )
                )
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"[Calendar] failed to auto-create event for booking {booking.id}: {e}")

        return {"status": "success", "transactionId": transaction_id}
    except Exception as e:
        session.rollback()
        # RECONCILIATION: money captured but booking not saved. Log loudly so
        # this can be reconciled manually against Braintree settlements.
        print(
            f"⚠️ RECONCILIATION NEEDED: charged transaction {transaction_id} "
            f"for user {current_user.id} / lawyer {body.lawyerId} but booking write "
            f"failed: {e}"
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
