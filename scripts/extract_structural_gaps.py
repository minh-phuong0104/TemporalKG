"""
Phase 3, Step 8: Use the trained model to predict structural gaps on the
CURRENT (final-year) graph.

PIPELINE:
    load graph -> load embeddings -> identify active nodes
    -> nearest-neighbor candidate shortlist (using EMBEDDINGS, cosine metric)
    -> build_pair_feature() for each candidate (using the SAME 193-dim
       feature function as training, from scripts/feature_utils.py)
    -> StandardScaler (the one fit during training, loaded from the saved
       model file -- never refit here)
    -> Logistic Regression -> probability -> rank -> Top-N structural gaps

IMPORTANT DISTINCTION (documented here because a reviewer will ask this):
Embeddings are used ONLY to shortlist plausible candidate pairs via nearest-
neighbor search (a cheap approximation to avoid scoring all ~53M possible
pairs). The actual prediction score comes from the trained classifier
operating on the richer 193-dim engineered feature (Hadamard + L1 + L2 +
cosine), NOT from raw embedding distance. These are two different, deliberate
uses of the same embeddings: one for cheap shortlisting, one for accurate
scoring.

Usage:
    python -m scripts.extract_structural_gaps
"""

import json
import pickle
from pathlib import Path
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import GRAPH_JSON_FILE, PROJECT_ROOT
from scripts.feature_utils import build_pair_feature, cosine_similarity
FINAL_YEAR = 2024
RECENT_WINDOW = 2  # "active" = has an edge within the most recent N years
EMBEDDING_FILE = PROJECT_ROOT / "data" / "embeddings" / f"emb_{FINAL_YEAR}.pkl"
MODEL_FILE = PROJECT_ROOT / "models" / "link_predictor.pkl"
OUTPUT_FILE = PROJECT_ROOT / "results" / "structural_gaps.json"

TOP_K_NEIGHBORS = 30   # candidate shortlist size per active node (raised from 15:
                       # graph has ~10k nodes, a wider shortlist reduces the risk
                       # of missing a strong candidate due to embedding noise)
TOP_N_GAPS = 50

CONFIDENCE_BANDS = [(0.9, "HIGH"), (0.75, "MEDIUM"), (0.0, "LOW")]


def confidence_label(p: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if p >= threshold:
            return label
    return "LOW"


def load_model():
    with open(MODEL_FILE, "rb") as f:
        saved = pickle.load(f)
    # Defensive: support both {"model":..., "scaler":...} and a bare model,
    # in case train_link_predictor.py's save format changes again later.
    if isinstance(saved, dict) and "model" in saved:
        return saved["model"], saved.get("scaler")
    return saved, None


def main():
    graph_data = json.loads(GRAPH_JSON_FILE.read_text(encoding="utf-8"))
    nodes, edges = graph_data["nodes"], graph_data["edges"]

    with open(EMBEDDING_FILE, "rb") as f:
        embeddings = pickle.load(f)
    model, scaler = load_model()

    node_ids = [n["id"] for n in nodes if n["id"] in embeddings]

    node_labels = {n["id"]: n.get("label", n["id"]) for n in nodes}


    def valid_entity(text: str) -> bool:
        if not text:
            return False
        text = text.lower().strip()
        words = text.split()

        blacklist = [
            "each",
            "more than",
            "number of",
            "case",
            "test case",
            "study",
            "results",
            "findings",
            "performance",
        ]

        if len(words) > 8:
            return False

        for x in blacklist:
            if text.startswith(x):
                return False

        return True


    node_ids = [
        n["id"]
        for n in nodes
        if n["id"] in embeddings and valid_entity(n.get("label", ""))
    ]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    matrix = np.array([embeddings[nid] for nid in node_ids])

    existing_pairs = {tuple(sorted([e["source"], e["target"]])) for e in edges}

    # --- Degree + common-neighbors lookups, for explainability fields ---
    adjacency: dict[str, set[str]] = {}
    for e in edges:
        adjacency.setdefault(e["source"], set()).add(e["target"])
        adjacency.setdefault(e["target"], set()).add(e["source"])
    degree = {n: len(adjacency.get(n, set())) for n in node_ids}

    def common_neighbors(a: str, b: str) -> int:
        return len(adjacency.get(a, set()) & adjacency.get(b, set()))

    # --- Active nodes: flexible recent-window definition (point 4 from review) ---
    active_cutoff = FINAL_YEAR - RECENT_WINDOW + 1
    active_nodes = set()
    for e in edges:
        if e["year"] >= active_cutoff:
            active_nodes.add(e["source"])
            active_nodes.add(e["target"])
    active_nodes = [n for n in active_nodes if n in id_to_idx]
    print(f"Active nodes (edge in {active_cutoff}-{FINAL_YEAR}): {len(active_nodes)} / {len(node_ids)}")

    # --- Candidate shortlisting via embedding nearest-neighbor search ---
    # NOTE: this ONLY shortlists candidates. It does NOT score them -- see
    # module docstring above.
    nn = NearestNeighbors(n_neighbors=min(TOP_K_NEIGHBORS + 1, len(node_ids)), metric="cosine")
    nn.fit(matrix)

    candidates = []
    seen_pairs = set()
    for node in active_nodes:
        idx = id_to_idx[node]
        distances, indices = nn.kneighbors(matrix[idx].reshape(1, -1))
        for j in indices[0]:
            neighbor = node_ids[j]
            if neighbor == node:
                continue
            pair = tuple(sorted([node, neighbor]))
            if pair in existing_pairs or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            candidates.append(pair)

    print(f"Candidate pairs to score: {len(candidates)}")
    if not candidates:
        print("⚠️  Không có candidate nào -- kiểm tra lại RECENT_WINDOW / TOP_K_NEIGHBORS.")
        return

    # --- Score candidates using the SAME feature function as training ---
    X_candidates = np.array([
        build_pair_feature(embeddings[a], embeddings[b]) for a, b in candidates
    ])
    if scaler is not None:
        X_candidates = scaler.transform(X_candidates)
    model_scores = model.predict_proba(X_candidates)[:, 1]

    # Baseline scores (raw cosine similarity) saved alongside, for comparison
    # in the report -- same idea as the classifier-vs-baseline comparison
    # already done in evaluate_link_predictor.py (point 10 from review).
    baseline_scores = np.array([cosine_similarity(embeddings[a], embeddings[b]) for a, b in candidates])

    ranked_idx = np.argsort(-model_scores)[:TOP_N_GAPS]

    gaps = []
    for rank, i in enumerate(ranked_idx, start=1):
        a, b = candidates[i]
        p = float(model_scores[i])
        gaps.append({
            "type": "structural_gap",
            "rank": rank,
            "node_a": a,
            "node_b": b,
            "label_a": node_labels.get(a, a),
            "label_b": node_labels.get(b, b),
            "probability": round(p, 4),
            "confidence": confidence_label(p),
            "cosine_similarity": round(float(baseline_scores[i]), 4),
            "degree_a": degree.get(a, 0),
            "degree_b": degree.get(b, 0),
            "common_neighbors": common_neighbors(a, b),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")

    # Baseline top-50 by cosine similarity alone, saved separately for the
    # Results section's model-vs-baseline comparison on the FINAL gap list
    # (distinct from the train/test AUC comparison already done earlier).
    baseline_ranked_idx = np.argsort(-baseline_scores)[:TOP_N_GAPS]
    baseline_gaps = []
    for rank, i in enumerate(baseline_ranked_idx, start=1):
        a, b = candidates[i]
        baseline_gaps.append({
            "rank": rank, "node_a": a, "node_b": b,
            "label_a": node_labels.get(a, a), "label_b": node_labels.get(b, b),
            "cosine_similarity": round(float(baseline_scores[i]), 4),
        })
    (PROJECT_ROOT / "results" / "structural_gaps_baseline_cosine.json").write_text(
        json.dumps(baseline_gaps, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✅ Top {len(gaps)} structural gaps saved to {OUTPUT_FILE}")
    print(f"💾 Baseline (cosine-only) top-{len(baseline_gaps)} saved for comparison.")
    print("\n--- Top 10 ---")
    for g in gaps[:10]:
        print(f"  #{g['rank']} {g['label_a']} <-> {g['label_b']}  "
              f"p={g['probability']} ({g['confidence']})  common_neighbors={g['common_neighbors']}")


if __name__ == "__main__":
    main()