"""
Linear Regression Solver
Solves using the Normal Equation: w = (XᵀX)⁻¹ Xᵀy
Shows every matrix operation step by step.
"""
import numpy as np
import plotly.graph_objects as go
from solvers.base import SolutionStep, SolutionResult, fig_to_b64


def solve_linear_regression(X: list, y: list, feature_names: list = None) -> SolutionResult:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(n_features)]

    result = SolutionResult(
        problem_type="Linear Regression",
        input_summary=f"{n_samples} samples, {n_features} feature(s): {', '.join(feature_names)}",
        related_topics=["Logistic Regression", "PCA", "Gradient Descent", "Ridge Regression"],
        interview_framing="Often asked as: 'Derive the closed-form solution for linear regression' or "
                          "'What are the assumptions of linear regression?'"
    )

    steps = []
    step = 1

    # Step 1 — Add bias column
    X_b = np.hstack([np.ones((n_samples, 1)), X])
    steps.append(SolutionStep(
        step_number=step,
        title="Add bias (intercept) column",
        calculation="X_b = [1 | X]  →  prepend a column of ones",
        result=X_b,
        explanation="We add a column of 1s so the model learns an intercept (bias) term. "
                    "Without it, the regression line is forced through the origin.",
        hint_1="We need an intercept term in our model.",
        hint_2="Adding a column of 1s lets the dot product learn a constant offset.",
        hint_3="X_b = np.hstack([np.ones((n,1)), X])",
    ))
    step += 1

    # Step 2 — Compute XᵀX
    XtX = X_b.T @ X_b
    steps.append(SolutionStep(
        step_number=step,
        title="Compute Xᵀ · X",
        calculation="XᵀX = X_bᵀ @ X_b",
        result=XtX,
        explanation="This matrix captures the relationship between every pair of features. "
                    "It's the denominator in the normal equation — tells us how 'spread out' X is.",
        hint_1="Matrix multiply X transposed with X.",
        hint_2="XᵀX has shape (n_features+1, n_features+1).",
        hint_3="XtX = X_b.T @ X_b",
    ))
    step += 1

    # Step 3 — Compute Xᵀy
    Xty = X_b.T @ y
    steps.append(SolutionStep(
        step_number=step,
        title="Compute Xᵀ · y",
        calculation="Xᵀy = X_bᵀ @ y",
        result=Xty,
        explanation="This vector captures how each feature correlates with the target. "
                    "It's the numerator — how much X 'pulls' toward y.",
        hint_1="Matrix multiply X transposed with the target vector y.",
        hint_2="Result is a vector of length n_features+1.",
        hint_3="Xty = X_b.T @ y",
    ))
    step += 1

    # Step 4 — Invert XᵀX
    XtX_inv = np.linalg.inv(XtX)
    steps.append(SolutionStep(
        step_number=step,
        title="Invert (XᵀX)⁻¹",
        calculation="(XᵀX)⁻¹ = np.linalg.inv(XᵀX)",
        result=XtX_inv,
        explanation="We invert XᵀX to 'divide' — this is the matrix equivalent of dividing both sides "
                    "of a scalar equation. If XᵀX isn't invertible, we use pseudoinverse instead.",
        hint_1="We need to invert the XᵀX matrix.",
        hint_2="Use np.linalg.inv() or np.linalg.pinv() if singular.",
        hint_3="XtX_inv = np.linalg.inv(XtX)",
    ))
    step += 1

    # Step 5 — Compute weights
    w = XtX_inv @ Xty
    steps.append(SolutionStep(
        step_number=step,
        title="Compute weights w = (XᵀX)⁻¹ Xᵀy",
        calculation="w = (XᵀX)⁻¹ · Xᵀy",
        result=w,
        explanation=f"These are our model parameters. w[0]={w[0]:.4f} is the intercept. "
                    + " ".join([f"w[{i+1}]={w[i+1]:.4f} is the coefficient for {feature_names[i]}."
                                for i in range(n_features)]),
        hint_1="Multiply (XᵀX)⁻¹ by Xᵀy.",
        hint_2="Result is a vector of weights, one per feature plus intercept.",
        hint_3="w = XtX_inv @ Xty",
    ))
    step += 1

    # Step 6 — Predictions
    y_pred = X_b @ w
    steps.append(SolutionStep(
        step_number=step,
        title="Compute predictions ŷ = X_b · w",
        calculation="ŷ = X_b @ w",
        result=y_pred,
        explanation="We apply our learned weights to each sample to get predicted values.",
        hint_1="Multiply the augmented X matrix by our weight vector.",
        hint_2="Each row of X_b dotted with w gives one prediction.",
        hint_3="y_pred = X_b @ w",
    ))
    step += 1

    # Step 7 — MSE and R²
    mse = float(np.mean((y - y_pred) ** 2))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    steps.append(SolutionStep(
        step_number=step,
        title="Evaluate: MSE and R²",
        calculation=f"MSE = mean((y - ŷ)²) = {mse:.4f}\nR² = 1 - SS_res/SS_tot = {r2:.4f}",
        result={"MSE": round(mse, 4), "R2": round(r2, 4)},
        explanation=f"MSE={mse:.4f} is the average squared error — lower is better. "
                    f"R²={r2:.4f} means the model explains {r2*100:.1f}% of variance in y. "
                    f"R²=1.0 is perfect; R²=0 means the model is no better than predicting the mean.",
        hint_1="MSE = mean of squared differences between actual and predicted.",
        hint_2="R² compares your model's error to a baseline that just predicts the mean.",
        hint_3="r2 = 1 - sum((y-ypred)**2) / sum((y-ymean)**2)",
    ))
    step += 1

    # Step 8 — Plot (only for single feature)
    if n_features == 1:
        fig = go.Figure()
        
        # Scatter plot of actual data
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=y,
            mode='markers',
            name='Actual',
            marker=dict(color='#185FA5', size=8)
        ))
        
        # Regression line
        x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_line = w[0] + w[1] * x_line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f"ŷ = {w[0]:.2f} + {w[1]:.2f}x",
            line=dict(color='#D85A30', width=2)
        ))
        
        fig.update_layout(
            title="Linear Regression Fit",
            xaxis_title=feature_names[0],
            yaxis_title="y",
            showlegend=True,
            template="plotly_white"
        )
        
        visual = fig_to_b64(fig)

        steps.append(SolutionStep(
            step_number=step,
            title="Visualize the regression line",
            calculation=f"ŷ = {w[0]:.4f} + {w[1]:.4f} · {feature_names[0]}",
            result="See plot",
            explanation="The red line is our learned regression function. Blue dots are actual data points. "
                        "A good fit means points cluster tightly around the line.",
            visual_b64=visual,
            hint_1="Plot actual points as a scatter, draw the regression line separately.",
        ))

    result.steps = steps
    result.final_answer = {
        "intercept": round(float(w[0]), 6),
        "coefficients": {feature_names[i]: round(float(w[i+1]), 6) for i in range(n_features)},
        "MSE": round(mse, 4),
        "R2": round(r2, 4),
        "equation": f"y = {w[0]:.4f} + " + " + ".join([f"{w[i+1]:.4f}·{feature_names[i]}" for i in range(n_features)])
    }

    return result
