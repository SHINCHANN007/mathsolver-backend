from fastapi import APIRouter, UploadFile, File, Header
from middleware.auth import verify_token
import pandas as pd
import io

router = APIRouter()


@router.post("/csv")
async def upload_csv(file: UploadFile = File(...), authorization: str = Header(...)):
    verify_token(authorization)

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Last column assumed to be target y; rest are features
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]

    X = df[feature_names].values.tolist()
    y = df[target_name].values.tolist()

    # Detect if classification or regression based on unique y values
    unique_y = list(set(y))
    is_classification = len(unique_y) <= 10 and all(
        isinstance(v, (int, float)) and float(v).is_integer() for v in unique_y
    )

    return {
        "feature_names": feature_names,
        "target_name": target_name,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "X": X,
        "y": y,
        "suggested_problems": (
            ["logistic_regression", "lda", "knn", "decision_tree"]
            if is_classification
            else ["linear_regression", "pca", "ridge_regression"]
        ),
        "preview": df.head(5).to_dict(orient="records"),
    }
