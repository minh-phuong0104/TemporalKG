import requests
import time
import json
import math
import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


from scripts.config import SEARCH_CONCEPT, YEAR_FROM, YEAR_TO

try:
    from scripts.config import SEMANTIC_SCHOLAR_API_KEY
except ImportError:
    SEMANTIC_SCHOLAR_API_KEY = None

try:
    from scripts.config import RAW_PAPERS_FILE_S2 as DEFAULT_OUTPUT
except ImportError:
    DEFAULT_OUTPUT = Path("data/raw/semanticscholar_papers.json")


BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,abstract,year,citationCount,referenceCount,externalIds,authors.name"

# Semantic Scholar: giới hạn 1 request/giây TÍNH CHUNG cho mọi endpoint dùng key này.
# Thực tế endpoint /search/bulk hay bị chặn burst chặt hơn mức docs ghi, nên để 2s
# cho an toàn — đỡ tốn thời gian chờ Retry-After (20-60s) mỗi lần dính 429.
MIN_REQUEST_INTERVAL = 2.0
_last_request_time = [0.0]


def rate_limited_get(url, **kwargs):
    elapsed = time.monotonic() - _last_request_time[0]
    wait = MIN_REQUEST_INTERVAL - elapsed
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, **kwargs)
    _last_request_time[0] = time.monotonic()
    return resp


# ============================================================
# CHECKPOINT
# ============================================================
def checkpoint_path_for(output: Path) -> Path:
    return output.with_suffix(".checkpoint.json")


def load_checkpoint(output: Path) -> dict:
    cp_path = checkpoint_path_for(output)
    if cp_path.exists():
        try:
            return json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            print(f" ⚠️ Không đọc được checkpoint cũ, bắt đầu tiến độ mới.")
    return {"year_state": {}}


def save_checkpoint(output: Path, checkpoint: dict) -> None:
    checkpoint_path_for(output).write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_existing_papers(output: Path) -> list[dict]:
    if output.exists():
        try:
            return json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            print(f" ⚠️ Không đọc được file cũ {output.name}, bắt đầu danh sách mới.")
    return []


def save_all_papers(output: Path, all_papers: list[dict]) -> None:
    output.write_text(json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# CÀO 1 NĂM, TỚI HẠN MỨC (cap) HIỆN TẠI CỦA NĂM ĐÓ
# Trả về True nếu năm này đã CẠN dữ liệu thật sự (không thể lấy thêm nữa dù tăng cap).
# ============================================================
def fetch_year_up_to_cap(concept, year, year_state, seen_ids, all_papers, output, checkpoint) -> bool:
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    cap = year_state["cap"]
    token = year_state["token"]
    count = year_state["count"]

    while count < cap:
        params = {"query": concept, "year": str(year), "fields": FIELDS}
        if token:
            params["token"] = token

        max_retries = 3
        resp = None
        gave_up = False

        for attempt in range(max_retries):
            try:
                resp = rate_limited_get(BASE_URL, params=params, headers=headers, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 20 * (attempt + 1)))
                    print(f" ⏳ Bị giới hạn tốc độ (429). Đợi {retry_after}s ({attempt+1}/{max_retries})...")
                    time.sleep(retry_after)
                    resp = None
                    continue
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f" ❌ Lỗi liên tục ở năm {year}: {e}. Tạm dừng năm này, checkpoint đã lưu.")
                    gave_up = True
                time.sleep(5)
        else:
            gave_up = True

        if gave_up or resp is None:
            # Không phải "cạn dữ liệu" — chỉ là lỗi tạm thời, thử lại ở vòng sau.
            return False

        data = resp.json()
        results = data.get("data", [])
        if not results:
            year_state["exhausted"] = True
            save_checkpoint(output, checkpoint)
            print(f" 🛑 Năm {year}: đã cạn dữ liệu thật sự ở mức {count} bài (API không còn gì để trả).")
            return True

        for work in results:
            work_id = work.get("paperId")
            if not work_id or work_id in seen_ids:
                continue
            abstract = work.get("abstract") or ""
            if not abstract or len(abstract.split()) < 20:
                continue
            seen_ids.add(work_id)
            all_papers.append({
                "id": work_id,
                "title": work.get("title"),
                "year": work.get("year"),
                "abstract": abstract,
                "citationCount": work.get("citationCount"),
                "referenceCount": work.get("referenceCount"),
                "doi": (work.get("externalIds") or {}).get("DOI"),
                "authors": [a.get("name") for a in (work.get("authors") or [])],
            })
            count += 1
            if count >= cap:
                break

        next_token = data.get("token")
        year_state["token"] = next_token
        year_state["count"] = count
        save_all_papers(output, all_papers)
        save_checkpoint(output, checkpoint)
        print(f"   💾 Năm {year}: {count}/{cap} (cap hiện tại) | Tổng toàn bộ: {len(all_papers)} bài")

        if not next_token:
            year_state["exhausted"] = True
            save_checkpoint(output, checkpoint)
            print(f" 🛑 Năm {year}: đã cạn dữ liệu thật sự ở mức {count} bài (hết token phân trang).")
            return True

        token = next_token

    return False  # đạt cap của vòng này, chưa chắc đã cạn


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a GUARANTEED total number of papers from Semantic Scholar, spread across years, with resume.")
    parser.add_argument("--concept", default=SEARCH_CONCEPT)
    parser.add_argument("--year-from", type=int, default=YEAR_FROM)
    parser.add_argument("--year-to", type=int, default=YEAR_TO)
    parser.add_argument("--target-total", type=int, default=15000, help="Tổng số bài MUỐN có, chắc chắn đạt được nếu dữ liệu cho phép.")
    parser.add_argument("--initial-per-year", type=int, default=None, help="Cap khởi điểm mỗi năm (mặc định = target/số năm).")
    parser.add_argument("--increment", type=int, default=500, help="Mỗi vòng thiếu thì tăng cap các năm chưa cạn thêm bao nhiêu.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not SEMANTIC_SCHOLAR_API_KEY:
        print(" ⚠️ Chưa có SEMANTIC_SCHOLAR_API_KEY trong scripts/config.py — tốc độ sẽ rất chậm.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_papers = load_existing_papers(args.output)
    seen_ids = {p["id"] for p in all_papers if p.get("id")}
    checkpoint = load_checkpoint(args.output)
    year_state_all = checkpoint.setdefault("year_state", {})

    years = list(range(args.year_from, args.year_to + 1))
    initial_cap = args.initial_per_year or math.ceil(args.target_total / len(years))

    # Khởi tạo / khôi phục trạng thái từng năm
    for year in years:
        key = str(year)
        if key not in year_state_all:
            count_from_disk = sum(1 for p in all_papers if p.get("year") == year)
            year_state_all[key] = {
                "cap": initial_cap,
                "token": None,
                "count": count_from_disk,
                "exhausted": False,
            }

    print(f"🚀 Mục tiêu: {args.target_total} bài | Khoảng năm: {args.year_from}-{args.year_to} | Từ khóa: '{args.concept}'")
    if all_papers:
        print(f"🔄 Đã có sẵn {len(all_papers)} bài từ lần chạy trước, sẽ tiếp tục chứ không cào lại.")
    print("=" * 50)

    try:
        round_num = 1
        while True:
            total = len(all_papers)
            if total >= args.target_total:
                print(f"\n🎉 ĐÃ ĐỦ MỤC TIÊU: {total}/{args.target_total} bài!")
                break

            print(f"\n===== VÒNG {round_num} | Hiện có {total}/{args.target_total} bài =====")
            for year in years:
                st = year_state_all[str(year)]
                if st["exhausted"] or st["count"] >= st["cap"]:
                    continue
                fetch_year_up_to_cap(args.concept, year, st, seen_ids, all_papers, args.output, checkpoint)
                if len(all_papers) >= args.target_total:
                    break

            total = len(all_papers)
            if total >= args.target_total:
                print(f"\n🎉 ĐÃ ĐỦ MỤC TIÊU: {total}/{args.target_total} bài!")
                break

            non_exhausted = [y for y in years if not year_state_all[str(y)]["exhausted"]]
            if not non_exhausted:
                print(f"\n⚠️ TẤT CẢ CÁC NĂM ĐÃ CẠN DỮ LIỆU. Chỉ lấy được {total}/{args.target_total} bài — "
                      f"không còn bài nào để cào thêm trong khoảng năm {args.year_from}-{args.year_to} với từ khóa '{args.concept}'. "
                      f"Cân nhắc mở rộng khoảng năm hoặc đổi từ khóa.")
                break

            for year in non_exhausted:
                year_state_all[str(year)]["cap"] += args.increment
            save_checkpoint(args.output, checkpoint)
            print(f" 📈 Còn thiếu {args.target_total - total} bài. Tăng cap thêm {args.increment} "
                  f"cho {len(non_exhausted)} năm chưa cạn, cào tiếp vòng {round_num + 1}...")
            round_num += 1
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n⏸️  Bị dừng (Ctrl+C). Đã lưu tiến độ — chạy lại đúng lệnh này để tiếp tục từ chỗ dừng.")
        save_all_papers(args.output, all_papers)
        save_checkpoint(args.output, checkpoint)


if __name__ == "__main__":
    main()