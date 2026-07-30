"""
Phase 3, Step 7: Evaluate the trained self-supervised link prediction model.

This script compares:

1. Logistic Regression (trained on Hadamard Node2Vec features)
2. Cosine Similarity baseline (no training)

Evaluation protocol
-------------------
Train:
    2020, 2021

Validation:
    2022

Test:
    2023

The decision threshold is selected ONLY on the validation split by
maximizing F1, then reused unchanged on the test split.

This prevents indirect test leakage.

Outputs
-------
results/
    evaluation.csv
    evaluation.md

results/figures/
    roc_curve.png
    pr_curve.png
    confusion_matrix.png
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT


# ============================================================
# Configuration
# ============================================================

FEATURES_FILE = PROJECT_ROOT / "data" / "training_pairs" / "features.npz"

MODEL_FILE = PROJECT_ROOT / "models" / "link_predictor.pkl"

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = RESULTS_DIR / "figures"

VAL_CUTOFFS = {2022}

TEST_CUTOFFS = {2023}

K_VALUES = [10, 20, 50]

THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)


# ============================================================
# Helper functions
# ============================================================

def split_by_cutoff(
    X,
    cosine_scores,
    labels,
    cutoff_years,
    selected_cutoffs,
):
    """
    Return only rows belonging to the requested cutoff years.
    """

    mask = np.isin(cutoff_years, list(selected_cutoffs))

    return (
        X[mask],
        cosine_scores[mask],
        labels[mask],
    )


def precision_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """
    Precision among the top-k highest scored pairs.
    """

    if len(scores) == 0:
        return float("nan")

    k = min(k, len(scores))

    ranked = np.argsort(-scores)

    top_k = ranked[:k]

    return float(labels[top_k].sum() / k)


def choose_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Search thresholds and maximize validation F1.
    """

    best_threshold = 0.5

    best_f1 = -1.0

    for threshold in THRESHOLD_GRID:

        predictions = (scores >= threshold).astype(int)

        if predictions.sum() == 0:
            continue

        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = current_f1

            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_scores(
    split_name: str,
    method_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
):
    """
    Evaluate one prediction method.

    Parameters
    ----------
    scores:
        Probability (model) or cosine similarity (baseline).

    threshold:
        Threshold already selected on validation.
    """

    predictions = (scores >= threshold).astype(int)

    metrics = {

        "Split": split_name,

        "Method": method_name,

        "Samples": len(labels),

        "Positive": int(labels.sum()),

        "Accuracy": accuracy_score(
            labels,
            predictions,
        ),

        "Precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            labels,
            scores,
        ),

        "Average Precision": average_precision_score(
            labels,
            scores,
        ),
    }

    return metrics


def print_metrics(metrics: dict):
    """
    Pretty console output.
    """

    print()

    print("=" * 60)

    print(
        f"{metrics['Method']}  |  {metrics['Split']}"
    )

    print("=" * 60)

    print(f"Samples           : {metrics['Samples']}")

    print(f"Positive samples  : {metrics['Positive']}")

    print(f"Accuracy          : {metrics['Accuracy']:.3f}")

    print(f"Precision         : {metrics['Precision']:.3f}")

    print(f"Recall            : {metrics['Recall']:.3f}")

    print(f"F1-score          : {metrics['F1']:.3f}")

    print(f"ROC-AUC           : {metrics['ROC-AUC']:.3f}")

    print(f"Average Precision : {metrics['Average Precision']:.3f}")
    
    # ============================================================
# Data loading
# ============================================================

def load_features():
    """
    Load feature matrix produced by build_features.py.
    """

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURES_FILE} not found. Run build_features.py first."
        )

    data = np.load(FEATURES_FILE)

    return (
        data["X"],
        data["cos_sim"],
        data["y"],
        data["cutoff_year"],
    )


def load_trained_model():
    """
    Load trained Logistic Regression model and fitted scaler.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Run train_link_predictor.py first."
        )

    with open(MODEL_FILE, "rb") as f:
        saved = pickle.load(f)

    return saved["model"], saved["scaler"]


# ============================================================
# Main evaluation
# ============================================================

def run_evaluation():

    X, cosine_scores, labels, cutoff_years = load_features()

    model, scaler = load_trained_model()

    (
        X_val,
        cosine_val,
        y_val,
    ) = split_by_cutoff(
        X,
        cosine_scores,
        labels,
        cutoff_years,
        VAL_CUTOFFS,
    )

    (
        X_test,
        cosine_test,
        y_test,
    ) = split_by_cutoff(
        X,
        cosine_scores,
        labels,
        cutoff_years,
        TEST_CUTOFFS,
    )

    if len(y_val) == 0:
        raise RuntimeError("Validation split is empty.")

    if len(y_test) == 0:
        raise RuntimeError("Test split is empty.")

    print("\nDataset summary")
    print("------------------------------")

    print(
        f"Validation : {len(y_val)} samples "
        f"({y_val.sum()} positive)"
    )

    print(
        f"Test       : {len(y_test)} samples "
        f"({y_test.sum()} positive)"
    )

    # --------------------------------------------------------
    # Apply the SAME scaler fitted during training
    # --------------------------------------------------------

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Logistic Regression probabilities
    # --------------------------------------------------------

    model_val_scores = model.predict_proba(
        X_val_scaled
    )[:, 1]

    model_test_scores = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    model_threshold, model_best_f1 = choose_best_threshold(
        model_val_scores,
        y_val,
    )

    cosine_threshold, cosine_best_f1 = choose_best_threshold(
        cosine_val,
        y_val,
    )

    print("\nThreshold selection")
    print("------------------------------")

    print(
        f"Model threshold  : {model_threshold:.2f}"
    )

    print(
        f"Validation F1    : {model_best_f1:.3f}"
    )

    print()

    print(
        f"Cosine threshold : {cosine_threshold:.2f}"
    )

    print(
        f"Validation F1    : {cosine_best_f1:.3f}"
    )

    # --------------------------------------------------------
    # Logistic Regression evaluation
    # --------------------------------------------------------

    val_metrics = evaluate_scores(
        split_name="Validation",
        method_name="Logistic Regression",
        scores=model_val_scores,
        labels=y_val,
        threshold=model_threshold,
    )

    test_metrics = evaluate_scores(
        split_name="Test",
        method_name="Logistic Regression",
        scores=model_test_scores,
        labels=y_test,
        threshold=model_threshold,
    )

    # --------------------------------------------------------
    # Cosine baseline
    # --------------------------------------------------------

    baseline_val_metrics = evaluate_scores(
        split_name="Validation",
        method_name="Cosine Similarity",
        scores=cosine_val,
        labels=y_val,
        threshold=cosine_threshold,
    )

    baseline_test_metrics = evaluate_scores(
        split_name="Test",
        method_name="Cosine Similarity",
        scores=cosine_test,
        labels=y_test,
        threshold=cosine_threshold,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print_metrics(val_metrics)

    print_metrics(test_metrics)

    print_metrics(baseline_val_metrics)

    print_metrics(baseline_test_metrics)

    # --------------------------------------------------------
    # Precision@K
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print("Precision@K (TEST)")

    print("=" * 60)

    precision_table = []

    for k in K_VALUES:

        model_precision = precision_at_k(
            model_test_scores,
            y_test,
            k,
        )

        baseline_precision = precision_at_k(
            cosine_test,
            y_test,
            k,
        )

        precision_table.append(
            {
                "k": k,
                "Model": model_precision,
                "Baseline": baseline_precision,
            }
        )

        print(
            f"P@{k:<2}   "
            f"Model={model_precision:.3f}   "
            f"Baseline={baseline_precision:.3f}"
        )

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "baseline_validation": baseline_val_metrics,
        "baseline_test": baseline_test_metrics,
        "precision_table": precision_table,
        "model_scores": model_test_scores,
        "baseline_scores": cosine_test,
        "labels": y_test,
        "model_threshold": model_threshold,
        "baseline_threshold": cosine_threshold,
    }
    # ============================================================
# Save reports
# ============================================================

def save_reports(results):
    """
    Save evaluation results as CSV and Markdown.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        results["validation"],
        results["test"],
        results["baseline_validation"],
        results["baseline_test"],
    ]

    df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "evaluation.csv"

    df.to_csv(csv_path, index=False)

    md_path = RESULTS_DIR / "evaluation.md"

    lines = [
        "# Link Prediction Evaluation",
        "",
        "## Classification Metrics",
        "",
        df.to_markdown(index=False),
        "",
        "## Precision@K (Test)",
        "",
        "| k | Logistic Regression | Cosine Baseline |",
        "|---|---|---|",
    ]

    for row in results["precision_table"]:
        lines.append(
            f"| {row['k']} | {row['Model']:.3f} | {row['Baseline']:.3f} |"
        )

    lines.append("")
    lines.append(
        f"Validation-selected threshold (model): "
        f"{results['model_threshold']:.2f}"
    )

    lines.append(
        f"Validation-selected threshold (baseline): "
        f"{results['baseline_threshold']:.2f}"
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()

    print(f"Saved CSV      -> {csv_path}")

    print(f"Saved Markdown -> {md_path}")


# ============================================================
# Plot figures
# ============================================================

def plot_figures(results):

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    labels = results["labels"]

    model_scores = results["model_scores"]

    baseline_scores = results["baseline_scores"]

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr_model, tpr_model, _ = roc_curve(
        labels,
        model_scores,
    )

    fpr_base, tpr_base, _ = roc_curve(
        labels,
        baseline_scores,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr_model,
        tpr_model,
        label=f"Logistic Regression (AUC={roc_auc_score(labels, model_scores):.3f})",
    )

    plt.plot(
        fpr_base,
        tpr_base,
        label=f"Cosine Baseline (AUC={roc_auc_score(labels, baseline_scores):.3f})",
    )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Precision-Recall Curve
    # --------------------------------------------------------

    precision_model, recall_model, _ = precision_recall_curve(
        labels,
        model_scores,
    )

    precision_base, recall_base, _ = precision_recall_curve(
        labels,
        baseline_scores,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        recall_model,
        precision_model,
        label=(
            f"Logistic Regression "
            f"(AP={average_precision_score(labels, model_scores):.3f})"
        ),
    )

    plt.plot(
        recall_base,
        precision_base,
        label=(
            f"Cosine Baseline "
            f"(AP={average_precision_score(labels, baseline_scores):.3f})"
        ),
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "pr_curve.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    predictions = (
        results["model_scores"] >= results["model_threshold"]
    ).astype(int)

    cm = confusion_matrix(
        labels,
        predictions,
    )

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title("Confusion Matrix (Test Set)")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    print()

    print(f"Saved ROC curve        -> {FIGURE_DIR / 'roc_curve.png'}")

    print(f"Saved PR curve         -> {FIGURE_DIR / 'pr_curve.png'}")

    print(f"Saved Confusion Matrix -> {FIGURE_DIR / 'confusion_matrix.png'}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print("Temporal Link Prediction Evaluation")

    print("=" * 70)

    results = run_evaluation()

    save_reports(results)

    plot_figures(results)

    print()

    print("=" * 70)

    print("Evaluation completed successfully.")

    print("=" * 70)


if __name__ == "__main__":

    main()