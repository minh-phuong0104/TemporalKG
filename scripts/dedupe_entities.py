"""
Find near-duplicate entities in the temporal KG that survived exact-match
canonicalization (e.g. "GPT-3" vs "GPT3" vs "gpt 3").

DESIGN NOTE (important): this version deliberately does NOT use transitive
clustering (no union-find). An earlier version did, and it badly over-merged
unrelated concepts (e.g. "deep learning", "LSTM", and "GPT-2" all ended up in
one 58-member cluster) because of the "chaining effect": if A~B and B~C are
each individually similar enough, transitive clustering merges A and C even
if they are not similar to each other at all. This version only ever merges
a DIRECT pair (an entity and its single nearest neighbor), and only when that
pair passes a high similarity threshold -- no multi-hop propagation.

Even so: for concept names, "textually similar" does not always mean "the
same real-world entity" (e.g. "deep learning" and "deep learning methods" are
arguably fine to merge; "biomedical NLP" and generic "NLP" may or may not be,
depending on how you want to treat domain-specific sub-fields). This script
still only PROPOSES merges -- manual review of results/entity_merge_candidates.md
remains mandatory before applying data/entity_cluster_mapping.json.

Usage:
    python -m scripts.dedupe_entities
"""

import json
import re
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from scripts.config import GRAPH_JSON_FILE, PROJECT_ROOT

# Raised from 0.80 -> 0.90: char n-gram cosine similarity runs high for any
# strings sharing common ML/NLP vocabulary words, so a "loose" threshold
# catches far too much. 0.90 still won't be perfect -- review the output.
SIMILARITY_THRESHOLD = 0.90

MERGE_MAPPING_FILE = PROJECT_ROOT / "data" / "entity_cluster_mapping.json"
REVIEW_FILE = PROJECT_ROOT / "results" / "entity_merge_candidates.md"

NUMBER_PATTERN = re.compile(r"\d+")

# Words so generic that "X <-> X + this word" pairs (e.g. "algorithm" vs
# "algorithms", or worse "network" vs "neural network") are NOT safe to
# auto-merge just because they're textually close -- require exact match
# after stripping only plurals, not these.
GENERIC_HEAD_WORDS = {
    "model", "models", "method", "methods", "algorithm", "algorithms",
    "technique", "techniques", "approach", "approaches", "system", "systems",
    "network", "networks", "architecture", "architectures",
}


def has_conflicting_numbers(a: str, b: str) -> bool:
    """True if both strings contain numbers AND those numbers differ --
    a strong signal these are different entities (GPT-2 vs GPT-3), not
    surface-form variants of the same one."""
    nums_a = set(NUMBER_PATTERN.findall(a))
    nums_b = set(NUMBER_PATTERN.findall(b))
    return bool(nums_a) and bool(nums_b) and nums_a != nums_b


def is_risky_generic_merge(a: str, b: str) -> bool:
    """Flag (for review, not auto-reject) pairs where the ONLY difference is
    one string being a bare generic word and the other a longer phrase
    containing it (e.g. "network" vs "neural network") -- these are the
    pattern most likely to be a false merge of genuinely different concepts."""
    words_a, words_b = set(a.split()), set(b.split())
    shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    return shorter.issubset(GENERIC_HEAD_WORDS) or (len(shorter) == 1 and next(iter(shorter)) in GENERIC_HEAD_WORDS)


def find_merge_candidates(entity_ids: list[str]):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(entity_ids)

    # Only look at each entity's single nearest neighbor -- no chaining,
    # no connected-component grouping.
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)

    mapping = {}
    review_rows = []

    for i, entity in enumerate(entity_ids):
        nearest_idx = indices[i][1] if indices[i][0] == i else indices[i][0]
        nearest_dist = distances[i][1] if indices[i][0] == i else distances[i][0]
        similarity = 1 - nearest_dist
        neighbor = entity_ids[nearest_idx]

        if similarity < SIMILARITY_THRESHOLD:
            continue
        if has_conflicting_numbers(entity, neighbor):
            continue

        # Always collapse the LONGER string toward the SHORTER one (heuristic:
        # shorter phrase is usually the more canonical/general form). Only
        # write a mapping FROM the longer string -- never remap the shorter
        # one -- this guarantees a single hop, no chains.
        if len(entity) == len(neighbor):
            continue  # ambiguous which is canonical; skip, let reviewer decide manually
        longer, shorter = (entity, neighbor) if len(entity) > len(neighbor) else (neighbor, entity)

        risky = is_risky_generic_merge(entity, neighbor)
        mapping[longer] = shorter
        review_rows.append((longer, shorter, round(similarity, 3), risky))

    return mapping, review_rows


def main():
    graph_data = json.loads(GRAPH_JSON_FILE.read_text(encoding="utf-8"))
    entity_ids = [node["id"] for node in graph_data["nodes"]]
    print(f"📊 Đang phân tích {len(entity_ids)} entity (threshold={SIMILARITY_THRESHOLD}, no chaining)...")

    mapping, review_rows = find_merge_candidates(entity_ids)

    MERGE_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    MERGE_MAPPING_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    risky_count = sum(1 for _, _, _, risky in review_rows if risky)
    review_rows.sort(key=lambda r: (-r[3], -r[2]))  # risky first, then by similarity

    lines = [
        "# Entity Merge Candidates (Manual Review Required)",
        "",
        f"Found {len(review_rows)} candidate PAIRS (direct nearest-neighbor only, "
        f"no transitive clustering; threshold={SIMILARITY_THRESHOLD}).",
        f"⚠️ {risky_count} pairs are flagged RISKY (one side is a bare generic word "
        f"like 'network' vs a longer phrase like 'neural network') -- check these first.",
        "",
        "**Each row: `longer_string -> shorter_string (similarity)`. If a merge is",
        "wrong, delete that key from `data/entity_cluster_mapping.json`.**",
        "",
        "| Longer | -> | Shorter | Similarity | Risky? |",
        "|---|---|---|---|---|",
    ]
    for longer, shorter, sim, risky in review_rows:
        flag = "⚠️ YES" if risky else ""
        lines.append(f"| {longer} | -> | {shorter} | {sim} | {flag} |")

    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ Tìm được {len(review_rows)} cặp entity nghi trùng lặp ({risky_count} cặp RISKY cần xem kỹ).")
    print(f"💾 Mapping: {MERGE_MAPPING_FILE}")
    print(f"📋 Review: {REVIEW_FILE}")
    print("\n⚠️  Đọc kỹ file review, đặc biệt các dòng RISKY, trước khi áp dụng mapping.")


if __name__ == "__main__":
    main()