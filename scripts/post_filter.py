"""
Post-filter: run this AFTER extract_triples.py.
Removes quadruples that slipped past the prompt's rules — small local models
like Gemma3 don't follow instructions with 100% reliability, so a code-level
safety net catches what the prompt misses.

Usage:
    python -m scripts.post_filter
"""

import json
import argparse
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    CLEAN_QUADRUPLES_FILE,
    QUADRUPLES_FILE,
    REJECTED_QUADRUPLES_FILE,
    YEAR_FROM,
    YEAR_TO,
)

# Relations too generic to represent a meaningful scientific claim
BANNED_RELATIONS = {
    "is", "has", "had", "includes", "is_a", "authored", "presented_at",
    "is_organized", "managed_by", "selected_by", "held",
    "affected_by", "moved_to", "increased_by", "increases_by",
    "reached", "sent_out_for", "kept_at", "removed",
}

# Bare single-word objects/subjects that carry almost no information on
# their own (e.g. "X was_popular --> methods" tells you nothing useful)
GENERIC_TERMS = {
    "methods", "models", "results", "techniques", "tools", "task", "tasks",
    "system", "systems", "approach", "approaches", "issues", "examples",
}

# Vague quantifier/determiner words that, when directly modifying a generic
# term, still carry almost no information (e.g. "various models", "several
# methods", "these techniques") -- catches cases GENERIC_TERMS alone misses
VAGUE_QUALIFIERS = {"various", "several", "these", "those", "many", "some", "certain"}

MAX_ENTITY_WORDS = 6  # entities longer than this are usually vague noun phrases,
                       # not clean scientific concepts (e.g. "extensible by
                       # researchers, simple for practitioners, and fast...")


def normalize_relation(value: str) -> str:
    """Collapse grammatical number variants before filtering.

    Examples:
    - are_evaluated_on -> is_evaluated_on
    - were_used_for -> was_used_for
    """
    relation = " ".join((value or "").strip().casefold().split())
    relation = relation.replace("-", "_").replace(" ", "_")
    relation = re.sub(r"^are_", "is_", relation)
    relation = re.sub(r"^were_", "was_", relation)
    return relation


def normalize_year(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_generic_phrase(text: str) -> bool:
    words = text.lower().split()
    if not words:
        return True
    if words[-1] in GENERIC_TERMS and (len(words) == 1 or words[0] in VAGUE_QUALIFIERS):
        return True
    return False


def classify_quadruple(q: dict) -> str:
    """Return 'valid' or a specific rejection reason -- used both for
    filtering and for the error-type breakdown needed in the Methodology
    section's extraction reliability analysis."""
    # Dùng ( ... or "" ) để nếu LLM trả về None, nó sẽ tự ép về chuỗi rỗng
    subject = (q.get("subject") or "").strip().lower()
    relation = normalize_relation(q.get("relation"))
    obj = (q.get("object") or "").strip().lower()
    year = normalize_year(q.get("year"))

    if not subject or not obj or not relation:
        return "missing_field"
    if year is None or not (YEAR_FROM <= year <= YEAR_TO):
        return "invalid_year"
    if relation in BANNED_RELATIONS:
        return "banned_relation"
    if is_generic_phrase(subject) or is_generic_phrase(obj):
        return "generic_entity"
    if len(subject.split()) > MAX_ENTITY_WORDS or len(obj.split()) > MAX_ENTITY_WORDS:
        return "entity_too_long"
    return "valid"


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-filter extracted quadruples.")
    parser.add_argument("--input", type=Path, default=QUADRUPLES_FILE)
    parser.add_argument("--output", type=Path, default=CLEAN_QUADRUPLES_FILE)
    parser.add_argument("--rejected-output", type=Path, default=REJECTED_QUADRUPLES_FILE)
    args = parser.parse_args()

    quadruples = json.loads(args.input.read_text(encoding="utf-8"))

    clean, rejected = [], []
    reasons = Counter()

    for q in quadruples:
        verdict = classify_quadruple(q)
        reasons[verdict] += 1
        if verdict == "valid":
            clean_q = dict(q)
            clean_q["relation"] = normalize_relation(clean_q.get("relation"))
            clean_q["year"] = normalize_year(clean_q.get("year"))
            clean.append(clean_q)
        else:
            rejected_q = dict(q)
            rejected_q["_rejection_reason"] = verdict  # kept only in the rejected file, for error analysis
            rejected.append(rejected_q)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    args.rejected_output.write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Kept:     {len(clean)} / {len(quadruples)} quadruples")
    print(f"Rejected: {len(rejected)} / {len(quadruples)} quadruples")
    print("\n--- Breakdown by rejection reason ---")
    for reason, count in reasons.most_common():
        if reason != "valid":
            pct = count / len(quadruples) * 100
            print(f"  {reason:20s}: {count:5d} ({pct:.1f}%)")
    print(f"\nClean quadruples saved to {args.output}")
    print(f"Rejected quadruples saved to {args.rejected_output}")
    print("\n>>> Keep the rejected file with its '_rejection_reason' field -- it's")
    print(">>> ready-made evidence for the Methodology section's error analysis.")


if __name__ == "__main__":
    main()
