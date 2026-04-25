"""
Solver dispatcher — detects problem type from user input and routes to the correct solver.
Adding a new solver = add one entry to SOLVER_MAP and one keyword to KEYWORD_MAP.
"""
from solvers.linear_regression import solve_linear_regression
from solvers.pca import solve_pca
from solvers.logistic_regression import solve_logistic_regression

SOLVER_MAP = {
    "linear_regression":    solve_linear_regression,
    "pca":                  solve_pca,
    "logistic_regression":  solve_logistic_regression,
    # Add new solvers here as you build them:
    # "lda":               solve_lda,
    # "knn":               solve_knn,
    # "kmeans":            solve_kmeans,
    # "naive_bayes":       solve_naive_bayes,
    # "decision_tree":     solve_decision_tree,
    # "backprop":          solve_backprop,
}

KEYWORD_MAP = {
    "linear_regression":   ["linear regression", "normal equation", "least squares", "ols"],
    "pca":                 ["pca", "principal component", "dimensionality reduction", "eigenvalue", "eigenvector"],
    "logistic_regression": ["logistic regression", "sigmoid", "binary classification", "cross entropy", "log loss"],
}

RELATED_TOPICS = {
    "linear_regression":   ["logistic_regression", "pca"],
    "pca":                 ["linear_regression", "logistic_regression"],
    "logistic_regression": ["linear_regression", "pca"],
}


def detect_problem_type(problem_text: str) -> str | None:
    text = problem_text.lower()
    for problem_type, keywords in KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            return problem_type
    return None


def dispatch_solver(problem_type: str, data: dict) -> dict:
    """
    data dict keys (all optional depending on solver):
      X              — feature matrix (list of lists or list)
      y              — target vector (list)
      feature_names  — list of strings
      n_components   — int (for PCA)
      lr             — float (learning rate for gradient descent solvers)
      epochs         — int
    """
    if problem_type not in SOLVER_MAP:
        return {"error": f"Solver for '{problem_type}' not yet implemented."}

    solver_fn = SOLVER_MAP[problem_type]
    X = data.get("X")
    y = data.get("y")
    feature_names = data.get("feature_names")

    try:
        if problem_type == "linear_regression":
            solution = solver_fn(X=X, y=y, feature_names=feature_names)

        elif problem_type == "pca":
            n_components = data.get("n_components", 2)
            solution = solver_fn(X=X, n_components=n_components, feature_names=feature_names)

        elif problem_type == "logistic_regression":
            lr = data.get("lr", 0.1)
            epochs = data.get("epochs", 100)
            solution = solver_fn(X=X, y=y, feature_names=feature_names, lr=lr, epochs=epochs)

        return solution.to_dict()

    except Exception as e:
        return {"error": str(e)}


def list_supported_problems() -> list[dict]:
    return [
        {"id": k, "keywords": KEYWORD_MAP.get(k, [])}
        for k in SOLVER_MAP
    ]
