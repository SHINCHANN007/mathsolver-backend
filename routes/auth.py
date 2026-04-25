import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


class AuthRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
def signup(req: AuthRequest):
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Supabase env not set")

    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    try:
        res = sb.auth.sign_up({
            "email": req.email,
            "password": req.password
        })

        # If user created, try to insert profile (non-blocking)
        if res.user and SUPABASE_SERVICE_KEY:
            try:
                sb2 = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
                sb2.table("profiles").insert({
                    "id": res.user.id,
                    "is_pro": False
                }).execute()
            except Exception as e:
                # Don’t crash signup if profile insert fails
                print("Profile insert failed:", e)

        return {"message": "Check your email to confirm signup."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
def login(req: AuthRequest):
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Supabase env not set")

    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    try:
        res = sb.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password
        })

        if not res.session:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {
            "access_token": res.session.access_token,
            "user_id": res.user.id,
            "email": res.user.email,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))