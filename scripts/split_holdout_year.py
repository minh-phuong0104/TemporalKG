"""
Tách papers_clean.json thành 2 phần:
- Tập train: các bài <= 2025 (dùng để build graph / train link predictor)
- Tập holdout: các bài năm 2026 (giữ riêng để dự đoán / đánh giá link prediction)

Chạy từ thư mục gốc project:
    python scripts/split_holdout_year.py
"""
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import CLEANED_DIR, CLEANED_PAPERS_FILE

HOLDOUT_YEAR = 2026
HOLDOUT_FILE = CLEANED_DIR / "papers_2026_holdout.json"


def main() -> None:
    if not CLEANED_PAPERS_FILE.exists():
        raise SystemExit(f"❌ Không tìm thấy {CLEANED_PAPERS_FILE}. Chạy clean_data.py trước.")

    papers = json.loads(CLEANED_PAPERS_FILE.read_text(encoding="utf-8"))
    print(f"📂 Đọc {len(papers)} bài từ {CLEANED_PAPERS_FILE.name}")

    train = [p for p in papers if p.get("year", 0) < HOLDOUT_YEAR]
    holdout = [p for p in papers if p.get("year", 0) == HOLDOUT_YEAR]

    # Ghi đè lại papers_clean.json chỉ còn <= 2025 (đây là input cho các bước sau: extract_triples, build_graph...)
    CLEANED_PAPERS_FILE.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    HOLDOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_FILE.write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Train (<= {HOLDOUT_YEAR - 1}): {len(train)} bài -> {CLEANED_PAPERS_FILE}")
    print(f"✅ Holdout ({HOLDOUT_YEAR}):     {len(holdout)} bài -> {HOLDOUT_FILE}")
    print("\n⚠️ Lưu ý: papers_clean.json đã bị ghi đè, chỉ còn dữ liệu <= 2025.")
    print("   File gốc đầy đủ (có cả 2026) vẫn còn nguyên trong data/raw/combined_papers.json nếu cần cào lại.")


if __name__ == "__main__":
    main()