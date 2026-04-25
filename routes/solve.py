from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel
from typing import Optional
from solvers.dispatcher import dispatch_solver, detect_problem_type, list_supported_problems
from middleware.auth import verify_token, check_usage_limit, log_solve

router = APIRouter()


class SolveRequest(BaseModel):
    problem_type: Optional[str] = None   # If None, auto-detect from problem_text
    problem_text: Optional[str] = None   # Natural language description
    X: list                              # Feature matrix
    y: Optional[list] = None             # Target (not needed for PCA)
    feature_names: Optional[list] = None
    n_components: Optional[int] = 2
    lr: Optional[float] = 0.1
    epochs: Optional[int] = 100


@router.post("/")
def solve(request: SolveRequest, authorization: str = Header(...)):
    # 1. Verify token
    user = verify_token(authorization)

    # 2. Detect problem type if not provided
    problem_type = request.problem_type
    if not problem_type and request.problem_text:
        problem_type = detect_problem_type(request.problem_text)
    if not problem_type:
        return {"error": "Could not detect problem type. Please specify problem_type explicitly.",
                "supported": list_supported_problems()}

    # 3. Check usage limit (free tier = 5/day)
    check_usage_limit(user["id"])

    # 4. Run solver
    data = {
        "X": request.X,
        "y": request.y,
        "feature_names": request.feature_names,
        "n_components": request.n_components,
        "lr": request.lr,
        "epochs": request.epochs,
    }
    result = dispatch_solver(problem_type, data)

    # 5. Log usage
    if "error" not in result:
        log_solve(user["id"], problem_type)

    return result


@router.get("/supported")
def get_supported():
    return {"problems": list_supported_problems()}
