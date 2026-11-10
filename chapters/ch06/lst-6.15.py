# --8<-- [start:lst-6.15]
# architecting-intelligence/src/06_durable/saga_activities.py
from temporalio import activity
from pydantic import BaseModel
import httpx
import hashlib

class FlightBooking(BaseModel):
    """Result of flight booking activity."""
    confirmation_code: str
    cost_usd: float
    airline: str

@activity.defn
async def book_flight(args: dict) -> FlightBooking:
    """
    Forward transaction: Book a flight.
    
    Idempotent: Uses deterministic idempotency key based on activity ID.
    """
    # Generate idempotency key
    idempotency_key = hashlib.sha256(
        f"{args['tenant_id']}:{args['order_id']}:book:{args['version']}".encode()
    ).hexdigest()[:16]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://airline-api.example.com/bookings",
            json={
                "origin": args["origin"],
                "destination": args["destination"],
                "date": args["date"],
                "idempotency_key": idempotency_key
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
    
    return FlightBooking(
        confirmation_code=data["confirmation"],
        cost_usd=data["price"],
        airline=data["airline"]
    )

@activity.defn
async def cancel_flight(args: dict) -> dict:
    """
    Compensation transaction: Cancel a flight booking.
    
    Idempotent: Canceling the same confirmation code twice returns success.
    """
    confirmation = args["confirmation_code"]
    
    idempotency_key = hashlib.sha256(
        f"{args['tenant_id']}:{confirmation}:cancel".encode()
    ).hexdigest()[:16]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://airline-api.example.com/cancellations",
            json={
                "confirmation_code": confirmation,
                "idempotency_key": idempotency_key
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
    
    return {
        "refund_amount": data.get("refund", 0.0),
        "status": data["status"]  # "refunded", "travel_credit", "non_refundable"
    }

# Similar pairs for hotel and car rental...
# --8<-- [end:lst-6.15]
