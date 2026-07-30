"""
Clean raw OpenAlex paper records before LLM extraction.

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

from scripts.config import CLEANED_PAPERS_FILE, RAW_PAPERS_FILE

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

        if not title or not abstract or not year:
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
            "cited_by_count": paper.get("cited_by_count", 0),
        })

    stats["output"] = len(clean)
    return clean, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw OpenAlex paper records.")
    parser.add_argument("--input", type=Path, default=RAW_PAPERS_FILE)
    parser.add_argument("--output", type=Path, default=CLEANED_PAPERS_FILE)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"❌ Input file not found: {args.input}")

    print(f"🧹 Đang đọc dữ liệu thô từ {args.input.name}...")
    papers = json.loads(args.input.read_text(encoding="utf-8"))
    clean, stats = clean_papers(papers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- KẾT QUẢ LÀM SẠCH DỮ LIỆU ---")
    print(f"📦 Input papers:    {stats['input']}")
    print(f"🗑️ Duplicates:      {stats['duplicates_removed']}")
    print(f"🗑️ Missing fields:  {stats['missing_required_fields']}")
    print(f"🗑️ Short abstracts: {stats['short_or_empty_abstract']}")
    print(f"✨ Clean papers:    {stats['output']}")
    print(f"💾 Saved to {args.output}")

    # Báo cáo thống kê năm (Cực kỳ quan trọng để kiểm tra tính cân bằng)
    print("\n Phân bố bài báo sạch theo năm:")
    if clean:
        years = [p["year"] for p in clean]
        counts = Counter(years)
        for y in sorted(counts.keys()):
            percentage = (counts[y] / len(clean)) * 100
            print(f"  - Năm {y}: {counts[y]} bài ({percentage:.1f}%)")
    else:
        print("  - Không có bài báo nào hợp lệ.")


if __name__ == "__main__":
    main()