"""
Auth middleware using Supabase.
Free tier: 5 solves/day tracked in Supabase.
Pro tier: unlimited, checked via user metadata.
"""
import os
from dotenv import load_dotenv


#load_dotenv()

#SUPABASE_URL = os.getenv("SUPABASE_URL")
#SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

from fastapi import HTTPException, Header
from supabase import create_client, Client
from datetime import date

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def verify_token(authorization: str = Header(...)) -> dict:
    """Extract and verify Supabase JWT. Returns user dict."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    sb = get_supabase()

    try:
        user_response = sb.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": user.id, "email": user.email, "metadata": user.user_metadata}
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")


def check_usage_limit(user_id: str) -> None:
    """
    For free users: allow max 5 solves per calendar day.
    Pro users: skip this check entirely.
    Raises HTTP 429 if limit exceeded.
    """
    sb = get_supabase()
    today = date.today().isoformat()

    # Check if user is pro
    profile = sb.table("profiles").select("is_pro").eq("id", user_id).single().execute()
    if profile.data and profile.data.get("is_pro"):
        return  # Pro users have unlimited access

    # Count today's solves
    usage = (
        sb.table("usage_log")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("date", today)
        .execute()
    )
    count = usage.count or 0

    if count >= 5:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Free tier limit reached (5 solves/day). Upgrade to Pro for unlimited access.",
                "upgrade_url": "/pricing"
            }
        )


def log_solve(user_id: str, problem_type: str) -> None:
    """Record a solve event for usage tracking."""
    sb = get_supabase()
    sb.table("usage_log").insert({
        "user_id": user_id,
        "problem_type": problem_type,
        "date": date.today().isoformat()
    }).execute()


