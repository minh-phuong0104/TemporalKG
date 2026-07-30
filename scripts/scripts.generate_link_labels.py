import json
import random
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
OUTPUT_FILE = PROJECT_ROOT / "data" / "training_pairs" / "pairs_labeled.csv"

CUTOFF_YEARS = [2020, 2021, 2022, 2023]
FINAL_YEAR = 2024
NEGATIVE_RATIO = 1  # 1 negative sampled per positive; raise to 2-3 if you
                     # want a harder/more realistic imbalanced setting later


def load_snapshot(year: int) -> dict:
    return json.loads((SNAPSHOT_DIR / f"snapshot_{year}.json").read_text(encoding="utf-8"))


def edge_pair_set(edges: list[dict]) -> set[tuple[str, str]]:
    """Undirected pair set: (A,B) and (B,A) treated as the same connection
    for the purpose of 'are these two concepts linked at all', regardless
    of which one was the grammatical subject in a given quadruple."""
    return {tuple(sorted([e["source"], e["target"]])) for e in edges if e["source"] != e["target"]}


def generate_labels_for_cutoff(cutoff: int, final_snapshot: dict) -> list[dict]:
    snap_t = load_snapshot(cutoff)
    nodes_t = snap_t["nodes"]  # only nodes that existed by this cutoff
    connected_at_t = edge_pair_set(snap_t["edges"])
    connected_by_final = edge_pair_set(final_snapshot["edges"])

    # Restrict "final" connections to pairs where BOTH nodes already
    # existed at T -- a pair involving a node that didn't exist yet at T
    # isn't a meaningful "gap" candidate for that cutoff.
    nodes_t_set = set(nodes_t)

    positives = []
    for pair in connected_by_final - connected_at_t:
        a, b = pair
        if a in nodes_t_set and b in nodes_t_set:
            positives.append(pair)

    print(f"  Cutoff {cutoff}: {len(positives)} positive pairs found.")

    # Negative sampling: random pairs from nodes_t that are NOT connected
    # at T and NEVER become connected by FINAL_YEAR either.
    negatives = []
    target_neg_count = len(positives) * NEGATIVE_RATIO
    attempts = 0
    max_attempts = target_neg_count * 50  # safety cap against infinite loop
    while len(negatives) < target_neg_count and attempts < max_attempts:
        a, b = random.sample(nodes_t, 2)
        pair = tuple(sorted([a, b]))
        attempts += 1
        if pair in connected_at_t or pair in connected_by_final:
            continue
        negatives.append(pair)

    print(f"  Cutoff {cutoff}: {len(negatives)} negative pairs sampled "
          f"({attempts} attempts).")

    rows = []
    for a, b in positives:
        rows.append({"node_a": a, "node_b": b, "cutoff_year": cutoff, "label": 1})
    for a, b in negatives:
        rows.append({"node_a": a, "node_b": b, "cutoff_year": cutoff, "label": 0})
    return rows


def main():
    final_snapshot = load_snapshot(FINAL_YEAR)

    all_rows = []
    for cutoff in CUTOFF_YEARS:
        rows = generate_labels_for_cutoff(cutoff, final_snapshot)
        all_rows.extend(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_a", "node_b", "cutoff_year", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    n_pos = sum(1 for r in all_rows if r["label"] == 1)
    n_neg = sum(1 for r in all_rows if r["label"] == 0)
    print(f"\nTotal: {len(all_rows)} labeled pairs ({n_pos} positive, {n_neg} negative)")
    print(f"Saved to {OUTPUT_FILE}")
    print("\n>>> Next step: generate_embeddings.py -- build a SEPARATE Node2Vec")
    print(">>> embedding for each cutoff snapshot (never share embeddings across")
    print(">>> cutoffs -- that would leak future graph structure into earlier rows).")


if __name__ == "__main__":
    main()