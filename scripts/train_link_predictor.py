"""
Phase 3, Step 5-6: Train the self-supervised link prediction model.

Split is by CUTOFF YEAR, not random -- this simulates the real deployment
scenario (train on past cutoffs, evaluate on a later, unseen cutoff), and
avoids leaking information across the temporal split.

    Train:      cutoffs 2020, 2021
    Validation: cutoff 2022
    Test:       cutoff 2023

Usage:
    python -m scripts.train_link_predictor
"""

import pickle
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT

FEATURES_FILE = PROJECT_ROOT / "data" / "training_pairs" / "features.npz"
MODEL_FILE = PROJECT_ROOT / "models" / "link_predictor.pkl"

TRAIN_CUTOFFS = {2020, 2021}
VAL_CUTOFFS = {2022}
TEST_CUTOFFS = {2023}

SEED = 42
EXPECTED_FEATURE_DIM = 193


def split_by_cutoff(X, cos_sim, y, cutoff_year, cutoffs):
    mask = np.isin(cutoff_year, list(cutoffs))
    return X[mask], cos_sim[mask], y[mask]


def main():
    data = np.load(FEATURES_FILE)
    X, cos_sim, y, cutoff_year = data["X"], data["cos_sim"], data["y"], data["cutoff_year"]

    print(f"Feature dimension: {X.shape[1]}")
    if X.shape[1] != EXPECTED_FEATURE_DIM:
        print(
            f"⚠ Warning: feature dimension = {X.shape[1]}, expected {EXPECTED_FEATURE_DIM}"
        )

    X_train, cos_train, y_train = split_by_cutoff(X, cos_sim, y, cutoff_year, TRAIN_CUTOFFS)
    X_val, cos_val, y_val = split_by_cutoff(X, cos_sim, y, cutoff_year, VAL_CUTOFFS)
    X_test, cos_test, y_test = split_by_cutoff(X, cos_sim, y, cutoff_year, TEST_CUTOFFS)

    print(f"Train: {len(y_train)} samples ({y_train.sum()} positive) -- cutoffs {sorted(TRAIN_CUTOFFS)}")
    print(f"Val:   {len(y_val)} samples ({y_val.sum()} positive) -- cutoffs {sorted(VAL_CUTOFFS)}")
    print(f"Test:  {len(y_test)} samples ({y_test.sum()} positive) -- cutoffs {sorted(TEST_CUTOFFS)}")
    print()
    print("Feature matrix")
    print("-------------------------")
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    if len(y_train) == 0 or len(np.unique(y_train)) < 2:
        raise SystemExit("❌ Train set rỗng hoặc chỉ có 1 class -- kiểm tra lại features.npz / cutoff.")

    # Fit StandardScaler ONLY on train -- never fit on val/test, that would
    # leak their distribution into the transform.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) else X_val
    X_test_scaled = scaler.transform(X_test) if len(X_test) else X_test
    print()
    print("Scaling completed.")

    model = LogisticRegression(
    max_iter=2000,
    C=0.1,
    class_weight="balanced",
    random_state=SEED
)
    model.fit(X_train_scaled, y_train)
    print()
    print(f"Iterations used: {model.n_iter_[0]}")

    train_acc = model.score(X_train_scaled, y_train)
    val_acc = model.score(X_val_scaled, y_val) if len(y_val) else float("nan")
    print()
    print("========================")
    print("Training finished")
    print("========================")
    print(f"Train Accuracy : {train_acc:.3f}")
    print(f"Val Accuracy   : {val_acc:.3f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        # save model AND scaler together -- evaluate_link_predictor.py must
        # apply the exact same fitted scaler, never refit one on test data
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_dimension": X.shape[1],
                "train_cutoffs": sorted(TRAIN_CUTOFFS),
                "val_cutoffs": sorted(VAL_CUTOFFS),
                "test_cutoffs": sorted(TEST_CUTOFFS),
                "seed": SEED,
            },
            f,
        )
    print(f"\n💾 Model + scaler saved to {MODEL_FILE}")
    print(">>> Next: python -m scripts.evaluate_link_predictor")


if __name__ == "__main__":
    main()
