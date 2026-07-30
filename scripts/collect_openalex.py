import requests
import time
import json
import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    OPENALEX_EMAIL,
    RAW_PAPERS_FILE,
    SEARCH_CONCEPT,
    YEAR_FROM,
    YEAR_TO,
)

# Kiểm tra xem có API Key trong config không
try:
    from scripts.config import OPENALEX_API_KEY
except ImportError:
    OPENALEX_API_KEY = None

BASE_URL = "https://api.openalex.org/works"

def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    position_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    ordered = [position_word[pos] for pos in sorted(position_word.keys())]
    return " ".join(ordered)

def fetch_papers_for_year(concept: str, year: int, limit_per_year: int):
    papers = []
    seen_ids = set()
    cursor = "*"
    headers = {"User-Agent": f"mailto:{OPENALEX_EMAIL}"}

    print(f"  👉 Đang lấy dữ liệu năm {year} (Mục tiêu: {limit_per_year} bài)...")

    while len(papers) < limit_per_year:
        params = {
            "search": concept,
            "filter": f"publication_year:{year},type:article",
            "per-page": 100,
            "cursor": cursor,
            "mailto": OPENALEX_EMAIL
        }
        
        # Nếu có API Key thì gửi kèm theo
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
                
                if resp.status_code == 429:
                    # Đọc thời gian chờ từ Server nếu có, mặc định tăng dần 20s, 40s
                    retry_after = int(resp.headers.get("Retry-After", 20 * (attempt + 1)))
                    print(f"    ⏳ IP bị khóa (429). Đợi {retry_after}s để mở khóa ({attempt+1}/{max_retries})...")
                    time.sleep(retry_after)
                    continue
                    
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"    ❌ Không thể lấy data năm {year} do IP đang bị block: {e}")
                    return papers
                time.sleep(5)
        else:
            print(f"    ⚠️ Năm {year} bị dừng do IP bị chặn liên tục. Chuyển sang thử năm tiếp theo...")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for work in results:
            work_id = work.get("id")
            if work_id in seen_ids:
                continue
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract or len(abstract.split()) < 20:
                continue
            seen_ids.add(work_id)
            papers.append({
                "id": work_id,
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "abstract": abstract,
                "cited_by_count": work.get("cited_by_count"),
            })
            if len(papers) >= limit_per_year:
                break

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        
        time.sleep(1.0) # Thời gian nghỉ giữa các trang

    print(f"  ✅ Hoàn thành năm {year}: Thu được {len(papers)} bài.")
    return papers

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paper records from OpenAlex year by year.")
    parser.add_argument("--concept", default=SEARCH_CONCEPT)
    parser.add_argument("--year-from", type=int, default=YEAR_FROM)
    parser.add_argument("--year-to", type=int, default=YEAR_TO)
    parser.add_argument("--per-year", type=int, default=300)
    parser.add_argument("--output", type=Path, default=RAW_PAPERS_FILE)
    args = parser.parse_args()

    all_papers = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Bắt đầu thu thập dữ liệu CHIA THEO NĂM ({args.year_from} -> {args.year_to})")
    print(f"Chỉ tiêu: {args.per_year} bài/năm | Từ khóa: '{args.concept}'\n" + "="*50)

    for year in range(args.year_from, args.year_to + 1):
        year_papers = fetch_papers_for_year(args.concept, year, args.per_year)
        
        if year_papers:
            all_papers.extend(year_papers)
            args.output.write_text(json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  💾 Đã lưu tổng cộng {len(all_papers)} bài vào {args.output.name}")
        
        time.sleep(2)
        print("-" * 30)

    print(f"\n🎉 HOÀN THÀNH TỔNG CỘNG: {len(all_papers)} bài báo đã được lưu an toàn!")

if __name__ == "__main__":
    main()