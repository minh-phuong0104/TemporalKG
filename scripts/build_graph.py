"""
Build a temporal knowledge graph from clean quadruples.

Reads outputs/quadruples_clean.json and writes:
- outputs/temporal_kg.json
- outputs/temporal_kg.graphml

Usage:
    python -m scripts.build_graph
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import networkx as nx

from scripts.config import CLEAN_QUADRUPLES_FILE, GRAPH_GRAPHML_FILE, GRAPH_JSON_FILE, PROJECT_ROOT

ENTITY_MAPPING_FILE = PROJECT_ROOT / "data" / "entity_cluster_mapping.json"

ALIASES = {
    "bert model": "bert",
    "bert-based model": "bert",
    "bidirectional encoder representations from transformers": "bert",
    "gpt model": "gpt",
    "transformer model": "transformer",
}

UPPER_VOCAB = {"bert", "gpt", "lstm", "cnn", "rnn", "nlp"}

STRIP_SUFFIXES = (" model", " method", " approach", " framework", " algorithm")
LEADING_ARTICLES = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def load_approved_merges() -> dict:
    """Load the manually-reviewed fuzzy-match mapping produced by
    scripts/dedupe_entities.py, if present. This is applied AFTER the
    rule-based canonicalize_entity() step, as a second pass."""
    if not ENTITY_MAPPING_FILE.exists():
        print("ℹ️  Không tìm thấy entity_cluster_mapping.json -- bỏ qua bước fuzzy merge "
              "(chỉ dùng canonicalization cơ bản). Chạy scripts/dedupe_entities.py trước "
              "nếu muốn áp dụng fuzzy merge đã duyệt.")
        return {}
    mapping = json.loads(ENTITY_MAPPING_FILE.read_text(encoding="utf-8"))
    print(f"✅ Đã tải {len(mapping)} entry từ entity_cluster_mapping.json (đã qua review thủ công).")
    return mapping


def canonicalize_entity(entity: str) -> str:
    """
    Returns a lowercase, alias-resolved canonical ID used ONLY to merge
    entities that refer to the same real-world concept regardless of surface
    form differences (capitalization, leading article, common suffix).

    IMPORTANT: this is the graph node ID (used for merging). It is NOT meant
    to be shown to a human reader -- see `pick_display_label` below for that.
    """
    text = re.sub(r"\s+", " ", entity or "").strip()
    text = text.strip(" .,:;\"'")
    text = LEADING_ARTICLES.sub("", text)  # "a BERT model" -> "BERT model"

    key = text.casefold()
    if key in ALIASES:
        return ALIASES[key]

    for suffix in STRIP_SUFFIXES:
        if key.endswith(suffix) and len(text.split()) > 1:
            text = text[: -len(suffix)].strip()
            key = text.casefold()
            break

    return key  # always lowercase -- this is what makes merging case-insensitive


def pick_display_label(canonical: str, surface_form_counts: Counter) -> str:
    """Choose a human-readable label for a canonical entity: the most
    frequently occurring original surface form in the dataset, or the
    uppercase acronym form for known abbreviations."""
    if canonical in UPPER_VOCAB:
        return canonical.upper()
    if surface_form_counts:
        return surface_form_counts.most_common(1)[0][0]
    return canonical


def build_temporal_graph(quadruples: list[dict], approved_merges: Optional[dict] = None) -> tuple[nx.MultiDiGraph, dict]:
    approved_merges = approved_merges or {}
    graph = nx.MultiDiGraph()

    # Pass 1: canonicalize everything, then apply reviewed fuzzy merges,
    # tracking which original surface forms map to each final canonical ID
    # (so we can later pick the best display label)
    surface_forms: dict[str, Counter] = {}
    skipped = 0
    prepared = []
    fuzzy_merges_applied = 0

    for q in quadruples:
        raw_subject = q.get("subject", "")
        raw_object = q.get("object", "")
        relation = re.sub(r"\s+", "_", q.get("relation", "").strip().lower())
        year = q.get("year")

        subject = canonicalize_entity(raw_subject)
        obj = canonicalize_entity(raw_object)

        # Second pass: apply manually-reviewed fuzzy merges on top of
        # rule-based canonicalization (e.g. "natural language processing"
        # -> "natural language processing (nlp)" if that pair was approved)
        if subject in approved_merges:
            subject = approved_merges[subject]
            fuzzy_merges_applied += 1
        if obj in approved_merges:
            obj = approved_merges[obj]
            fuzzy_merges_applied += 1

        if not subject or not obj or not relation or year is None:
            skipped += 1
            continue

        surface_forms.setdefault(subject, Counter())[raw_subject.strip()] += 1
        surface_forms.setdefault(obj, Counter())[raw_object.strip()] += 1

        prepared.append((subject, obj, relation, int(year), str(q.get("paper_id", ""))))

    # Pass 2: build the graph using canonical IDs as nodes, with a
    # human-readable "label" attribute chosen from the most common surface form
    for idx, (subject, obj, relation, year, paper_id) in enumerate(prepared):
        for node_id in (subject, obj):
            if node_id not in graph:
                graph.add_node(node_id, label=pick_display_label(node_id, surface_forms[node_id]))

        graph.add_edge(
            subject,
            obj,
            key=f"{relation}:{idx}",
            relation=relation,
            year=year,
            timestamp=year,
            paper_id=paper_id,
        )

    stats = {
        "input_quadruples": len(quadruples),
        "skipped_missing_field": skipped,
        "unique_canonical_entities": len(surface_forms),
        "merged_surface_forms": sum(1 for c in surface_forms.values() if len(c) > 1),
        "fuzzy_merges_applied": fuzzy_merges_applied,
    }
    return graph, stats


def graph_to_json(graph: nx.MultiDiGraph) -> dict:
    nodes = [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    edges = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edges.append({
            "source": source,
            "target": target,
            "key": key,
            **attrs,
        })
    return {"nodes": nodes, "edges": edges}


def print_summary(graph: nx.MultiDiGraph, stats: dict) -> None:
    relation_counts = Counter(data["relation"] for _, _, data in graph.edges(data=True))
    year_counts = Counter(data["year"] for _, _, data in graph.edges(data=True))
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Top relations: {relation_counts.most_common(10)}")
    print(f"Years: {sorted(year_counts.items())}")
    print("\n--- Entity normalization stats ---")
    print(f"Input quadruples:              {stats['input_quadruples']}")
    print(f"Skipped (missing field):       {stats['skipped_missing_field']}")
    print(f"Unique canonical entities:     {stats['unique_canonical_entities']}")
    print(f"Entities with >1 surface form: {stats['merged_surface_forms']} "
          f"(these were successfully merged, e.g. 'NLP' + 'nlp' + 'Nlp')")
    print(f"Fuzzy merges applied:          {stats['fuzzy_merges_applied']} "
          f"(from reviewed entity_cluster_mapping.json)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NetworkX temporal KG.")
    parser.add_argument("--input", type=Path, default=CLEAN_QUADRUPLES_FILE)
    parser.add_argument("--json-output", type=Path, default=GRAPH_JSON_FILE)
    parser.add_argument("--graphml-output", type=Path, default=GRAPH_GRAPHML_FILE)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    quadruples = json.loads(args.input.read_text(encoding="utf-8"))
    approved_merges = load_approved_merges()
    graph, stats = build_temporal_graph(quadruples, approved_merges)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(graph_to_json(graph), ensure_ascii=False, indent=2), encoding="utf-8")
    nx.write_graphml(graph, args.graphml_output)

    print_summary(graph, stats)
    print(f"\nSaved JSON to {args.json_output}")
    print(f"Saved GraphML to {args.graphml_output}")


if __name__ == "__main__":
    main()
