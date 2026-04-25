import os, hmac, hashlib, json
from fastapi import APIRouter, Request, Header, HTTPException
from pydantic import BaseModel
import razorpay
from supabase import create_client
from middleware.auth import verify_token

router = APIRouter()

RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
SUPABASE_URL        = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

PRO_AMOUNT_PAISE = 9900   # ₹99 in paise


class CreateOrderRequest(BaseModel):
    plan: str = "pro_monthly"


@router.post("/create-order")
def create_order(request: CreateOrderRequest, authorization: str = Header(...)):
    """Create a Razorpay order. Frontend uses the returned order_id to open payment modal."""
    user = verify_token(authorization)
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

    order = client.order.create({
        "amount": PRO_AMOUNT_PAISE,
        "currency": "INR",
        "receipt": f"mathsolver_{user['id'][:8]}",
        "notes": {"user_id": user["id"], "plan": request.plan}
    })

    return {
        "order_id": order["id"],
        "amount": PRO_AMOUNT_PAISE,
        "currency": "INR",
        "key_id": RAZORPAY_KEY_ID,
    }


@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay calls this after payment.
    Verifies signature, then upgrades user to pro in Supabase.
    Set this URL in Razorpay Dashboard → Settings → Webhooks.
    """
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    # Verify signature
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = json.loads(body)
    event = payload.get("event")

    if event == "payment.captured":
        payment = payload["payload"]["payment"]["entity"]
        notes = payment.get("notes", {})
        user_id = notes.get("user_id")

        if user_id:
            sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            # Upgrade user to pro
            sb.table("profiles").upsert({
                "id": user_id,
                "is_pro": True,
                "razorpay_payment_id": payment["id"]
            }).execute()

    return {"status": "ok"}
