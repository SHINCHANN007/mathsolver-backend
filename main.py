from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import solve, generate, upload, payments, auth
import uvicorn

app = FastAPI(title="MathSolver API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,     prefix="/auth",     tags=["auth"])
app.include_router(solve.router,    prefix="/solve",    tags=["solve"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(upload.router,   prefix="/upload",   tags=["upload"])
app.include_router(payments.router, prefix="/payments", tags=["payments"])

@app.get("/")
def root():
    return {"status": "MathSolver API running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
