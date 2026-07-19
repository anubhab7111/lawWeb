"""
Booking/payment endpoints using the Braintree Python SDK (Sandbox),
ported from the old Express server/src/routes/bookings.ts.
Credentials come strictly from settings (.env) — no hardcoded fallbacks.
"""

from functools import lru_cache
from typing import Optional

import braintree
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
from app.db.engine import get_session
from app.db.models import Booking, BookingStatus

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
def checkout(body: CheckoutRequest, session: Session = Depends(get_session)):
    """Charge the nonce received from the client and record the booking."""
    try:
        result = get_gateway().transaction.sale(
            {
                "amount": str(body.amount),
                "payment_method_nonce": body.paymentMethodNonce,
                "options": {"submit_for_settlement": True},
            }
        )

        if result.is_success:
            booking = Booking(
                user_id=body.userId,
                lawyer_id=body.lawyerId,
                amount=body.amount,
                status=BookingStatus.confirmed,
                transaction_id=result.transaction.id,
            )
            session.add(booking)
            session.commit()

            print(f"✅ Success: Payment settled for User {body.userId}")
            return {"status": "success", "transactionId": result.transaction.id}

        print(f"❌ Braintree Transaction Failed: {result.message}")
        return JSONResponse(
            status_code=400, content={"status": "error", "message": result.message}
        )
    except Exception as e:
        print(f"Checkout Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error during checkout"},
        )


@router.get("/user-bookings/{user_id}")
def user_bookings(user_id: str, session: Session = Depends(get_session)):
    """Confirmed appointments for a user, newest first."""
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
