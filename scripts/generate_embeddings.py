"""
Phase 3, Step 3: Generate Node2Vec embeddings SEPARATELY for each temporal
snapshot. This is a critical leakage-control point: an embedding trained on
snapshot(T) must only ever "see" edges with year <= T. We NEVER train one
embedding on the full/final graph and reuse it for earlier cutoffs.

Answers to the 5 review points raised in chat:
  1. Separate Node2Vec per snapshot -- yes, one full training run per file
     in data/snapshots/, no sharing of embeddings across cutoffs.
  2. Dimension -- 64. Kept modest on purpose: the labeled training set is
     only ~3,182 samples (Ngày 42-43 output), so a high-dimensional feature
     space (e.g. 128) risks overfitting the downstream classifier more than
     it risks underfitting from too few dimensions.
  3. Random seed -- fixed at 42, passed to Node2Vec's random walk generator
     AND to Word2Vec's training, so results are reproducible run-to-run.
  4. Disconnected nodes -- not actually possible in our data: build_snapshots.py
     only adds a node to a snapshot if it appears in at least one edge with
     year <= T, so by construction every node in every snapshot has degree >= 1
     and Node2Vec's random walks can always traverse from it.
  5. Saved as one file per cutoff year (emb_2020.pkl, emb_2021.pkl, ...),
     not one shared embeddings.pkl.

Usage:
    python -m scripts.generate_embeddings
"""

import json
import pickle
from pathlib import Path
import sys

import networkx as nx
from node2vec import Node2Vec

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

SEED = 42
DIMENSIONS = 64
WALK_LENGTH = 30
NUM_WALKS = 10
WINDOW_SIZE = 5


def build_networkx_graph(snapshot: dict) -> nx.Graph:
    """Undirected simple graph for embedding purposes -- consistent with
    generate_link_labels.py, which also treats connectivity as undirected
    (a link between A and B is a link, regardless of which one was the
    grammatical subject in the original quadruple)."""
    g = nx.Graph()
    g.add_nodes_from(snapshot["nodes"])
    for e in snapshot["edges"]:
        g.add_edge(e["source"], e["target"])
    return g


def train_embedding(graph: nx.Graph) -> dict:
    node2vec = Node2Vec(
        graph,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        seed=SEED,
        workers=1,  # workers=1 required for reproducibility with a fixed seed
                    # (multi-worker random walk generation is non-deterministic)
        quiet=True,
    )
    model = node2vec.fit(window=WINDOW_SIZE, min_count=1, seed=SEED)

    embeddings = {}
    for node in graph.nodes():
        embeddings[node] = model.wv[node]
    return embeddings


def main():
    if not SNAPSHOT_DIR.exists():
        raise SystemExit(f"❌ {SNAPSHOT_DIR} không tồn tại -- chạy build_snapshots.py trước.")

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_files = sorted(SNAPSHOT_DIR.glob("snapshot_*.json"))

    if not snapshot_files:
        raise SystemExit(f"❌ Không tìm thấy snapshot nào trong {SNAPSHOT_DIR}")

    for snap_path in snapshot_files:
        snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
        year = snapshot["cutoff_year"]

        print(f"Training Node2Vec for snapshot {year} "
              f"({len(snapshot['nodes'])} nodes, {len(snapshot['edges'])} edges)...")

        graph = build_networkx_graph(snapshot)
        embeddings = train_embedding(graph)

        out_path = EMBEDDING_DIR / f"emb_{year}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(embeddings, f)

        print(f"  ✅ Saved {len(embeddings)} vectors (dim={DIMENSIONS}) -> {out_path.name}")

    print(f"\nDone. Embeddings saved to {EMBEDDING_DIR}")
    print(">>> Next step: use these embeddings to build features for each")
    print(">>> labeled pair in data/training_pairs/pairs_labeled.csv, matching")
    print(">>> each row's cutoff_year to the embedding trained on that SAME year.")


if __name__ == "__main__":
    main()