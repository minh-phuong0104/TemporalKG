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


def normalize_text(value: str) -> str:
    """Xóa khoảng trắng thừa và ký tự xuống dòng."""
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str) -> str:
    """Chuẩn hóa tiêu đề về chữ thường để so sánh chống trùng lặp."""
    return normalize_text(value).casefold()


def paper_key(paper: dict) -> str:
    """Tạo khóa duy nhất (Unique Key) cho mỗi bài báo để lọc trùng."""
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
    }

    for paper in papers:
        title = normalize_text(paper.get("title", ""))
        abstract = normalize_text(paper.get("abstract", ""))
        year = paper.get("year")

        if not title or not abstract or year is None:
            stats["missing_required_fields"] += 1
            continue

        if len(abstract.split()) < MIN_ABSTRACT_WORDS:
            stats["short_or_empty_abstract"] += 1
            continue

        key = paper_key(paper)
        if key in seen:
            stats["duplicates_removed"] += 1
            continue

        seen.add(key)

        clean.append({
            "id": normalize_text(str(paper.get("id", ""))) or title,
            "title": title,
            "year": int(year),
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
        raise SystemExit(f"❌ Input file not found: {args.input}")

    print(f"🧹 Reading raw Semantic Scholar papers from:")
    print(f"   {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        papers = json.load(f)

    clean, stats = clean_papers(papers)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print("\n========== CLEANING SUMMARY ==========")
    print(f"Input papers:        {stats['input']}")
    print(f"Duplicates removed:  {stats['duplicates_removed']}")
    print(f"Missing fields:      {stats['missing_required_fields']}")
    print(f"Short abstracts:     {stats['short_or_empty_abstract']}")
    print(f"Clean papers:        {stats['output']}")
    print(f"Saved to:            {args.output}")

    print("\nPaper distribution by year:")
    if clean:
        years = [p["year"] for p in clean]
        counts = Counter(years)

        for y in sorted(counts):
            pct = counts[y] * 100 / len(clean)
            print(f"  {y}: {counts[y]} papers ({pct:.1f}%)")
    else:
        print("  No valid papers found.")


if __name__ == "__main__":
    main()