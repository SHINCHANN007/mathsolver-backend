"""
Logistic Regression Solver
Method: Gradient Descent on Binary Cross-Entropy Loss
Shows sigmoid, loss, gradient, weight update — every iteration.
"""
import numpy as np
import matplotlib.pyplot as plt
from solvers.base import SolutionStep, SolutionResult, fig_to_b64


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def solve_logistic_regression(
    X: list, y: list,
    feature_names: list = None,
    lr: float = 0.1,
    epochs: int = 100,
    show_every: int = 10
) -> SolutionResult:

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(n_features)]

    result = SolutionResult(
        problem_type="Logistic Regression",
        input_summary=f"{n_samples} samples, {n_features} feature(s), binary classification (0/1)",
        related_topics=["Linear Regression", "Sigmoid Function", "Gradient Descent",
                        "Binary Cross-Entropy", "SVM", "Neural Networks"],
        interview_framing="Commonly asked: 'Why do we use sigmoid in logistic regression?' "
                          "or 'Derive the gradient of cross-entropy loss.'"
    )

    steps = []
    step = 1

    # Step 1 — Add bias
    X_b = np.hstack([np.ones((n_samples, 1)), X])
    steps.append(SolutionStep(
        step_number=step,
        title="Add bias column and initialise weights",
        calculation="X_b = [1 | X],  w = zeros(n_features + 1)",
        result={"X_b shape": list(X_b.shape), "initial weights": [0.0] * (n_features + 1)},
        explanation="Same as linear regression — we add a bias column. "
                    "Weights start at zero so gradient descent has a neutral starting point.",
        hint_1="We need an intercept, so prepend a column of ones.",
        hint_2="Initialize all weights to 0.",
        hint_3="w = np.zeros(n_features + 1)",
    ))
    step += 1
    w = np.zeros(n_features + 1)

    # Step 2 — Sigmoid explanation
    z_demo = np.array([-3, -1, 0, 1, 3])
    sig_demo = sigmoid(z_demo)
    steps.append(SolutionStep(
        step_number=step,
        title="Understand the sigmoid function σ(z) = 1/(1+e⁻ᶻ)",
        calculation="σ(z) = 1 / (1 + exp(-z))",
        result={str(round(z, 1)): round(float(s), 4) for z, s in zip(z_demo, sig_demo)},
        explanation="Sigmoid squashes any real number into (0, 1), giving us a probability. "
                    "σ(0)=0.5 is the decision boundary. "
                    "If σ(z) > 0.5 → predict class 1, else class 0.",
        hint_1="Sigmoid maps any number to a value between 0 and 1.",
        hint_2="The output represents P(y=1 | x).",
        hint_3="def sigmoid(z): return 1 / (1 + np.exp(-z))",
    ))

    # Sigmoid plot
    fig, ax = plt.subplots(figsize=(6, 3))
    z_range = np.linspace(-6, 6, 200)
    ax.plot(z_range, sigmoid(z_range), color="#185FA5", linewidth=2)
    ax.axhline(0.5, color="#D85A30", linestyle="--", alpha=0.7, label="Decision boundary (0.5)")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("z = w·x")
    ax.set_ylabel("σ(z)")
    ax.set_title("Sigmoid Function")
    ax.legend()
    steps[-1].visual_b64 = fig_to_b64(fig)
    step += 1

    # Step 3 — Training loop
    loss_history = []
    weight_snapshots = {}

    for epoch in range(epochs):
        z = X_b @ w
        y_pred = sigmoid(z)
        loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
        gradient = (1 / n_samples) * X_b.T @ (y_pred - y)
        w = w - lr * gradient
        loss_history.append(loss)

        if epoch in [0, show_every - 1, epochs // 2, epochs - 1]:
            weight_snapshots[epoch] = {
                "weights": np.round(w, 4).tolist(),
                "loss": round(float(loss), 6),
                "gradient": np.round(gradient, 4).tolist()
            }

    # Step 4 — Show gradient descent mechanics
    snap_epoch = list(weight_snapshots.keys())[0]
    snap = weight_snapshots[snap_epoch]
    steps.append(SolutionStep(
        step_number=step,
        title="Gradient descent weight update (first iteration shown)",
        calculation=(
            "z = X_b @ w\n"
            "ŷ = σ(z)\n"
            "L = -mean(y·log(ŷ) + (1-y)·log(1-ŷ))   ← Binary Cross-Entropy\n"
            "∇L = (1/n) · X_bᵀ · (ŷ - y)\n"
            "w_new = w - α · ∇L"
        ),
        result={"after epoch 0": weight_snapshots[0]},
        explanation="Binary cross-entropy penalises confident wrong predictions heavily. "
                    f"Learning rate α={lr} controls how big each step is. "
                    "Each epoch: forward pass (compute ŷ) → loss → gradient → update weights.",
        hint_1="Gradient of cross-entropy loss w.r.t. w is (1/n)·Xᵀ·(ŷ - y).",
        hint_2="Subtract learning_rate × gradient from weights.",
        hint_3="gradient = (1/n) * X_b.T @ (y_pred - y)\nw -= lr * gradient",
    ))
    step += 1

    # Step 5 — Loss curve
    steps.append(SolutionStep(
        step_number=step,
        title=f"Training complete: loss over {epochs} epochs",
        calculation=f"Final loss = {loss_history[-1]:.6f}",
        result={"initial_loss": round(loss_history[0], 4), "final_loss": round(loss_history[-1], 4),
                "weight_snapshots": weight_snapshots},
        explanation=f"Loss dropped from {loss_history[0]:.4f} → {loss_history[-1]:.4f}. "
                    "A decreasing loss curve confirms the model is learning. "
                    "If it oscillates, the learning rate is too high. If it barely moves, it's too low.",
        hint_1="Track loss every epoch to verify learning.",
        hint_2="Loss should monotonically decrease for a reasonable learning rate.",
    ))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(loss_history, color="#185FA5", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss Curve")
    ax.set_ylim(bottom=0)
    steps[-1].visual_b64 = fig_to_b64(fig)
    step += 1

    # Step 6 — Final predictions and accuracy
    z_final = X_b @ w
    probs = sigmoid(z_final)
    preds = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(preds == y))

    steps.append(SolutionStep(
        step_number=step,
        title="Final predictions and accuracy",
        calculation="ŷ = σ(X_b @ w)\npred_class = 1 if ŷ >= 0.5 else 0\naccuracy = mean(pred_class == y)",
        result={"probabilities": np.round(probs, 3).tolist(),
                "predicted_classes": preds.tolist(),
                "actual_classes": y.astype(int).tolist(),
                "accuracy": f"{accuracy*100:.1f}%"},
        explanation=f"Model accuracy: {accuracy*100:.1f}%. "
                    "Each probability above 0.5 is classified as class 1. "
                    "For imbalanced datasets, also check precision and recall.",
        hint_1="Apply sigmoid to the final dot product to get probabilities.",
        hint_2="Threshold at 0.5 to get class predictions.",
        hint_3="preds = (sigmoid(X_b @ w) >= 0.5).astype(int)",
    ))

    # Decision boundary plot for single feature
    if n_features == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X[:, 0][y == 0], y[y == 0], color="#185FA5", label="Class 0", zorder=5)
        ax.scatter(X[:, 0][y == 1], y[y == 1], color="#D85A30", label="Class 1", zorder=5)
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
        x_b_line = np.column_stack([np.ones(200), x_line])
        prob_line = sigmoid(x_b_line @ w)
        ax.plot(x_line, prob_line, color="#639922", linewidth=2, label="P(y=1|x)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel("P(y=1)")
        ax.set_title("Logistic Regression — Sigmoid Fit")
        ax.legend()
        step += 1
        steps.append(SolutionStep(
            step_number=step,
            title="Visualize decision boundary",
            calculation="P(y=1|x) = σ(w₀ + w₁·x)",
            result="See plot",
            explanation="The S-curve shows the probability of class 1 for each x. "
                        "Where the curve crosses 0.5 is the decision boundary.",
            visual_b64=fig_to_b64(fig),
        ))

    result.steps = steps
    result.final_answer = {
        "intercept": round(float(w[0]), 6),
        "coefficients": {feature_names[i]: round(float(w[i+1]), 6) for i in range(n_features)},
        "final_loss": round(loss_history[-1], 6),
        "accuracy": f"{accuracy*100:.1f}%",
        "equation": f"P(y=1) = σ({w[0]:.4f} + " + " + ".join(
            [f"{w[i+1]:.4f}·{feature_names[i]}" for i in range(n_features)]) + ")"
    }

    return result
