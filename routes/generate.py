from fastapi import APIRouter, Header
from pydantic import BaseModel
from middleware.auth import verify_token
import numpy as np
import random

router = APIRouter()


class GenerateRequest(BaseModel):
    problem_type: str
    difficulty: str = "beginner"   # beginner | intermediate | exam


def _random_X_y(n_samples, n_features, classification=False, seed=None):
    rng = np.random.RandomState(seed)
    X = np.round(rng.randn(n_samples, n_features) * 5 + 10, 2).tolist()
    if classification:
        y = rng.randint(0, 2, size=n_samples).tolist()
    else:
        w_true = rng.randn(n_features)
        noise = rng.randn(n_samples) * 0.5
        y = (np.array(X) @ w_true + noise).round(2).tolist()
    return X, y


DIFFICULTY_CONFIG = {
    "beginner":     {"n_samples": 6,  "n_features": 1, "epochs": 50},
    "intermediate": {"n_samples": 20, "n_features": 2, "epochs": 100},
    "exam":         {"n_samples": 50, "n_features": 3, "epochs": 200},
}


@router.post("/")
def generate_problem(request: GenerateRequest, authorization: str = Header(...)):
    verify_token(authorization)
    cfg = DIFFICULTY_CONFIG.get(request.difficulty, DIFFICULTY_CONFIG["beginner"])
    seed = random.randint(1, 9999)

    if request.problem_type in ("linear_regression", "pca"):
        X, y = _random_X_y(cfg["n_samples"], cfg["n_features"], classification=False, seed=seed)
    elif request.problem_type == "logistic_regression":
        X, y = _random_X_y(cfg["n_samples"], cfg["n_features"], classification=True, seed=seed)
    else:
        return {"error": f"Generator not yet defined for {request.problem_type}"}

    feature_names = [f"x{i+1}" for i in range(cfg["n_features"])]

    return {
        "problem_type": request.problem_type,
        "difficulty": request.difficulty,
        "seed": seed,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "n_components": 2,
        "epochs": cfg["epochs"],
        "prompt": f"Solve {request.problem_type.replace('_', ' ').title()} on the given dataset "
                  f"({cfg['n_samples']} samples, {cfg['n_features']} feature(s)). "
                  f"Show all steps."
    }
