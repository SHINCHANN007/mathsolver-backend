"""
PCA Solver — Principal Component Analysis
Method: Covariance matrix → Eigendecomposition → Project data
"""
import numpy as np
#import matplotlib.pyplot as plt
from solvers.base import SolutionStep, SolutionResult, fig_to_b64


def solve_pca(X: list, n_components: int = 2, feature_names: list = None) -> SolutionResult:
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    n_components = min(n_components, n_features)
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(n_features)]

    result = SolutionResult(
        problem_type="PCA (Principal Component Analysis)",
        input_summary=f"{n_samples} samples × {n_features} features, reducing to {n_components} component(s)",
        related_topics=["LDA", "SVD", "Eigenvalues & Eigenvectors", "Covariance Matrix", "t-SNE"],
        interview_framing="Common interview question: 'Explain PCA step by step' or "
                          "'What is the difference between PCA and LDA?'"
    )

    steps = []
    step = 1

    # Step 1 — Mean center
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    steps.append(SolutionStep(
        step_number=step,
        title="Mean-center the data",
        calculation=f"mean = {np.round(mean, 4).tolist()}\nX_centered = X - mean",
        result=X_centered,
        explanation="PCA finds directions of maximum variance. To do that correctly, "
                    "we first shift the data so its centroid is at the origin. "
                    "This removes the effect of the mean and lets us focus purely on spread.",
        hint_1="Compute the mean of each feature column.",
        hint_2="Subtract the mean vector from every row of X.",
        hint_3="X_centered = X - X.mean(axis=0)",
    ))
    step += 1

    # Step 2 — Covariance matrix
    cov = np.cov(X_centered.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    steps.append(SolutionStep(
        step_number=step,
        title="Compute covariance matrix",
        calculation="C = (1/(n-1)) · X_centeredᵀ · X_centered",
        result=cov,
        explanation="The covariance matrix tells us how each pair of features varies together. "
                    "A large off-diagonal value means two features move together — they're redundant. "
                    "PCA will find axes that remove this redundancy.",
        hint_1="Covariance matrix is symmetric, shape (n_features × n_features).",
        hint_2="Use np.cov(X_centered.T) — note the transpose.",
        hint_3="cov = np.cov(X_centered.T)",
    ))
    step += 1

    # Step 3 — Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    steps.append(SolutionStep(
        step_number=step,
        title="Eigendecomposition of covariance matrix",
        calculation="C · v = λ · v  →  eigenvalues λ, eigenvectors v",
        result={"eigenvalues": np.round(eigenvalues, 4).tolist(),
                "eigenvectors (columns)": np.round(eigenvectors, 4).tolist()},
        explanation="Eigenvectors are the principal components — the new axes of maximum variance. "
                    "Eigenvalues tell us HOW MUCH variance each axis captures. "
                    "We sort them largest-to-smallest so PC1 always captures the most variance.",
        hint_1="Use np.linalg.eigh() for symmetric matrices (more stable than eig).",
        hint_2="Sort eigenvalues descending and reorder eigenvectors the same way.",
        hint_3="eigenvalues, eigenvectors = np.linalg.eigh(cov)\nidx = np.argsort(eigenvalues)[::-1]",
    ))
    step += 1

    # Step 4 — Explained variance
    total_var = np.sum(eigenvalues)
    explained = eigenvalues / total_var * 100
    cumulative = np.cumsum(explained)

    steps.append(SolutionStep(
        step_number=step,
        title="Compute explained variance ratio",
        calculation="explained[i] = eigenvalue[i] / sum(eigenvalues) × 100%",
        result={f"PC{i+1}": f"{explained[i]:.2f}% (cumulative: {cumulative[i]:.2f}%)"
                for i in range(len(eigenvalues))},
        explanation=f"PC1 alone explains {explained[0]:.1f}% of total variance. "
                    f"The first {n_components} component(s) together explain {cumulative[n_components-1]:.1f}%. "
                    "This tells you how much information you're keeping after dimensionality reduction.",
        hint_1="Divide each eigenvalue by the sum of all eigenvalues.",
        hint_2="Multiply by 100 for percentage. Cumsum gives cumulative explained variance.",
        hint_3="explained = eigenvalues / eigenvalues.sum() * 100",
    ))

    # Scree plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(1, len(eigenvalues)+1), explained, color="#185FA5", alpha=0.7, label="Individual")
    ax.plot(range(1, len(eigenvalues)+1), cumulative, "o-", color="#D85A30", label="Cumulative")
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot")
    ax.legend()
    steps[-1].visual_b64 = fig_to_b64(fig)
    step += 1

    # Step 5 — Select top components
    W = eigenvectors[:, :n_components]
    steps.append(SolutionStep(
        step_number=step,
        title=f"Select top {n_components} eigenvector(s) as projection matrix W",
        calculation=f"W = eigenvectors[:, :{n_components}]  →  shape {W.shape}",
        result=np.round(W, 4),
        explanation=f"We take only the top {n_components} eigenvectors. "
                    "These define our new lower-dimensional space. "
                    "Each column of W is a direction in the original feature space.",
        hint_1=f"Take the first {n_components} columns from the sorted eigenvector matrix.",
        hint_2="W has shape (n_features, n_components).",
        hint_3=f"W = eigenvectors[:, :{n_components}]",
    ))
    step += 1

    # Step 6 — Project data
    X_projected = X_centered @ W
    steps.append(SolutionStep(
        step_number=step,
        title="Project data into new space",
        calculation="X_projected = X_centered @ W",
        result=np.round(X_projected, 4),
        explanation="Each data point is now represented in the new PC space. "
                    "We've reduced dimensionality while preserving maximum variance. "
                    f"Original: {n_features} features → Now: {n_components} component(s).",
        hint_1="Multiply mean-centered X by the projection matrix W.",
        hint_2="Result shape: (n_samples, n_components).",
        hint_3="X_projected = X_centered @ W",
    ))
    step += 1

    # Step 7 — 2D plot if applicable
    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X_projected[:, 0], X_projected[:, 1], color="#185FA5", alpha=0.7, s=60)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
        ax.set_title("Data projected onto first 2 Principal Components")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        fig.tight_layout()
        visual = fig_to_b64(fig)
        steps.append(SolutionStep(
            step_number=step,
            title="Visualize projected data (PC1 vs PC2)",
            calculation="scatter(X_projected[:,0], X_projected[:,1])",
            result="See plot",
            explanation="Each point is your original data sample now living in PC space. "
                        "Clusters in this plot reveal natural groupings in your data. "
                        "The axes are the directions of maximum variance.",
            visual_b64=visual,
        ))

    result.steps = steps
    result.final_answer = {
        "n_components": n_components,
        "explained_variance_pct": [round(float(e), 2) for e in explained[:n_components]],
        "total_variance_retained_pct": round(float(cumulative[n_components-1]), 2),
        "projection_matrix_W": np.round(W, 4).tolist(),
        "projected_data_shape": list(X_projected.shape),
    }

    return result
