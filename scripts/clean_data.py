"""
Clean raw Semantic Scholar paper records before LLM extraction.

This step is intentionally separate from post_filter.py:
- clean_data.py cleans paper records before extraction
- post_filter.py cleans extracted quadruples after extraction

Usage:
    python -m scripts.clean_data
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import CLEANED_PAPERS_FILE, RAW_PAPERS_FILE_S2

MIN_ABSTRACT_WORDS = 20
MAX_YEAR = 2025   # Chỉ giữ dữ liệu lịch sử


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str) -> str:
    return normalize_text(value).casefold()


def paper_key(paper: dict) -> str:
    paper_id = normalize_text(str(paper.get("id", "")))
    if paper_id:
        return f"id:{paper_id}"
    return f"title:{normalize_title(paper.get('title', ''))}"


def clean_papers(papers: list[dict]) -> tuple[list[dict], dict]:
    clean = []
    seen = set()

    stats = {
        "input": len(papers),
        "duplicates_removed": 0,
        "missing_required_fields": 0,
        "short_or_empty_abstract": 0,
        "future_papers_removed": 0,
    }

    for paper in papers:

        title = normalize_text(paper.get("title", ""))
        abstract = normalize_text(paper.get("abstract", ""))
        year = paper.get("year")

        # Thiếu thông tin bắt buộc
        if not title or not abstract or year is None:
            stats["missing_required_fields"] += 1
            continue

        try:
            year = int(year)
        except Exception:
            stats["missing_required_fields"] += 1
            continue

        # Chỉ dùng dữ liệu lịch sử
        if year > MAX_YEAR:
            stats["future_papers_removed"] += 1
            continue

        # Abstract quá ngắn
        if len(abstract.split()) < MIN_ABSTRACT_WORDS:
            stats["short_or_empty_abstract"] += 1
            continue

        # Trùng bài
        key = paper_key(paper)
        if key in seen:
            stats["duplicates_removed"] += 1
            continue

        seen.add(key)

        clean.append({
            "id": normalize_text(str(paper.get("id", ""))) or title,
            "title": title,
            "year": year,
            "abstract": abstract,
            "cited_by_count": paper.get(
                "citationCount",
                paper.get("cited_by_count", 0)
            ),
        })

    stats["output"] = len(clean)

    return clean, stats


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Clean raw Semantic Scholar paper records."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_PAPERS_FILE_S2,
        help="Semantic Scholar raw paper file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=CLEANED_PAPERS_FILE,
        help="Output cleaned paper file",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    print(f"Reading Semantic Scholar papers from:")
    print(args.input)

    with open(args.input, "r", encoding="utf-8") as f:
        papers = json.load(f)

    clean, stats = clean_papers(papers)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print("\n========== CLEANING SUMMARY ==========")
    print(f"Input papers:         {stats['input']}")
    print(f"Duplicates removed:   {stats['duplicates_removed']}")
    print(f"Missing fields:       {stats['missing_required_fields']}")
    print(f"Short abstracts:      {stats['short_or_empty_abstract']}")
    print(f"Future papers:        {stats['future_papers_removed']}")
    print(f"Clean papers:         {stats['output']}")
    print(f"Saved to:             {args.output}")

    print("\nPaper distribution by year:")

    if clean:
        counts = Counter(p["year"] for p in clean)

        for year in sorted(counts):
            pct = counts[year] * 100 / len(clean)
            print(f"  {year}: {counts[year]} papers ({pct:.1f}%)")
    else:
        print("  No valid papers found.")


if __name__ == "__main__":
    main()