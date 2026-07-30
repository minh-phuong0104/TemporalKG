"""
Shared feature construction for node-pair features -- used by BOTH
build_features.py (training) and extract_structural_gaps.py (inference).

CRITICAL: this is the SINGLE SOURCE OF TRUTH for how a pair of embeddings
becomes a feature vector. build_features.py must import build_pair_feature()
from here instead of computing it inline, and extract_structural_gaps.py
must do the same -- otherwise the two scripts can silently drift apart
(same output shape, different meaning per dimension), which is worse than
a crash because nothing will look obviously wrong.

Feature = concat([hadamard, l1_distance, l2_distance, cosine_similarity])
        = 64 + 64 + 64 + 1 = 193 dimensions
(confirmed against the actual build_features.py implementation)

Feature key in features.npz is "X" (not "X_hadamard") -- make sure
train_link_predictor.py and evaluate_link_predictor.py load this key.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_pair_feature(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """Returns the 193-dim feature vector:
    [hadamard product (64), L1 distance (64), L2 distance (64), cosine similarity (1)]."""
    hadamard = vec_a * vec_b
    l1 = np.abs(vec_a - vec_b)
    l2 = (vec_a - vec_b) ** 2
    cos = cosine_similarity(vec_a, vec_b)
    return np.concatenate([hadamard, l1, l2, np.array([cos])])