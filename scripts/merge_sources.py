"""
Gộp 2 nguồn dữ liệu thô (OpenAlex + Semantic Scholar) thành 1 file duy nhất,
khử trùng lặp trước khi đưa vào clean_data.py.

Chạy từ thư mục gốc project:
    python scripts/merge_sources.py

Kết quả: data/raw/combined_papers.json
"""
import json
import re
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import RAW_DIR, RAW_PAPERS_FILE

# File Semantic Scholar: dùng RAW_PAPERS_FILE_S2 nếu đã khai trong config,
# nếu chưa thì fallback về đường dẫn mặc định mà collect_semanticscholar_papers.py dùng.
try:
    from scripts.config import RAW_PAPERS_FILE_S2
except ImportError:
    RAW_PAPERS_FILE_S2 = RAW_DIR / "semanticscholar_papers.json"

OUTPUT_FILE = RAW_DIR / "combined_papers.json"


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f" ⚠️ Không tìm thấy file: {path} — bỏ qua nguồn này.")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f" 📂 Đọc {len(data)} bài từ {path.name}")
        return data
    except Exception as e:
        print(f" ❌ Lỗi đọc {path}: {e}")
        return []


def normalize_title(title: str) -> str:
    """Chuẩn hóa tiêu đề để so khớp trùng lặp: lowercase, bỏ dấu câu, bỏ khoảng trắng thừa."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_doi(doi) -> str:
    if not doi:
        return ""
    d = str(doi).lower().strip()
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return d


def dedup_key(paper: dict) -> str:
    """Ưu tiên khớp theo DOI (chính xác nhất), nếu không có DOI thì khớp theo tiêu đề chuẩn hóa."""
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = normalize_title(paper.get("title"))
    return f"title:{title}" if title else f"id:{paper.get('id')}"


def better_record(a: dict, b: dict) -> dict:
    """Khi trùng, giữ bản có abstract dài hơn (thường đầy đủ/chất lượng hơn)."""
    len_a = len(a.get("abstract") or "")
    len_b = len(b.get("abstract") or "")
    return a if len_a >= len_b else b


def merge_sources() -> None:
    openalex_papers = load_json(RAW_PAPERS_FILE)
    for p in openalex_papers:
        p["source"] = "openalex"

    s2_papers = load_json(RAW_PAPERS_FILE_S2)
    for p in s2_papers:
        p["source"] = "semantic_scholar"

    all_raw = openalex_papers + s2_papers
    print(f"\n🔗 Tổng số bài trước khi khử trùng: {len(all_raw)}")

    merged: dict[str, dict] = {}
    no_title_no_doi = 0

    for paper in all_raw:
        if not paper.get("title") and not paper.get("doi"):
            no_title_no_doi += 1
            continue
        key = dedup_key(paper)
        if key in merged:
            merged[key] = better_record(merged[key], paper)
        else:
            merged[key] = paper

    combined = list(merged.values())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    dup_count = len(all_raw) - len(combined) - no_title_no_doi
    print(f" 🗑️  Loại trùng lặp: {dup_count} bài")
    if no_title_no_doi:
        print(f" 🗑️  Loại vì thiếu cả title lẫn DOI: {no_title_no_doi} bài")
    print(f" ✅ Đã lưu {len(combined)} bài vào: {OUTPUT_FILE}")

    by_source = {}
    for p in combined:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    print(f" 📊 Theo nguồn (sau khử trùng, ưu tiên bản abstract dài hơn khi trùng): {by_source}")


if __name__ == "__main__":
    merge_sources()