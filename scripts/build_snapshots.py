

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import GRAPH_JSON_FILE, PROJECT_ROOT

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"

# Cutoff years to generate. Adjust based on your actual data range (2018-2024).
# We need at least one "future" year beyond each cutoff to generate positive
# labels, so the last cutoff should leave room (e.g. don't use 2024 as a
# cutoff -- there'd be no future data left to check against).
CUTOFF_YEARS = [2020, 2021, 2022, 2023]
FINAL_YEAR = 2024  # the most recent year in the dataset -- used as the
                    # "ground truth future" against every cutoff


def load_edges():
    data = json.loads(GRAPH_JSON_FILE.read_text(encoding="utf-8"))
    return data["edges"]


def build_snapshot(edges: list[dict], cutoff: int) -> dict:
    """A node 'exists' at cutoff T if it appears in at least one edge with
    year <= T (we don't have a separate node-creation-year field, so this
    is the natural proxy)."""
    snap_edges = [e for e in edges if e["year"] <= cutoff]
    nodes = set()
    for e in snap_edges:
        nodes.add(e["source"])
        nodes.add(e["target"])
    return {"cutoff_year": cutoff, "nodes": sorted(nodes), "edges": snap_edges}


def main():
    edges = load_edges()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for cutoff in CUTOFF_YEARS + [FINAL_YEAR]:
        snapshot = build_snapshot(edges, cutoff)
        out_path = SNAPSHOT_DIR / f"snapshot_{cutoff}.json"
        out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Snapshot {cutoff}: {len(snapshot['nodes'])} nodes, {len(snapshot['edges'])} edges -> {out_path.name}")

    print(f"\nSaved {len(CUTOFF_YEARS) + 1} snapshots to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()