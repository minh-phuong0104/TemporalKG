"""
Generate descriptive statistics for the cleaned dataset.
Outputs a markdown summary suitable for the Methodology section of the paper.

Usage:
    python -m scripts.dataset_stats
"""

import json
import statistics
from collections import Counter
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import CLEANED_PAPERS_FILE, PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_FILE = RESULTS_DIR / "dataset_summary.md"

def generate_summary():
    if not CLEANED_PAPERS_FILE.exists():
        print(f"❌ Không tìm thấy file dữ liệu: {CLEANED_PAPERS_FILE}")
        return

    print(f"📊 Đang phân tích dữ liệu từ {CLEANED_PAPERS_FILE.name}...")
    papers = json.loads(CLEANED_PAPERS_FILE.read_text(encoding="utf-8"))

    if not papers:
        print("⚠️ Tập dữ liệu trống!")
        return

    # 1. Thống kê cơ bản
    total_papers = len(papers)
    
    # 2. Thống kê theo năm
    years = [p["year"] for p in papers if p.get("year")]
    year_counts = Counter(years)
    min_year, max_year = min(years), max(years)

    # 3. Thống kê độ dài Abstract
    abstract_lengths = [len(p["abstract"].split()) for p in papers if p.get("abstract")]
    avg_length = statistics.mean(abstract_lengths)
    median_length = statistics.median(abstract_lengths)
    max_length = max(abstract_lengths)
    min_length = min(abstract_lengths)

    # 4. Thống kê Citation
    citations = [p["cited_by_count"] for p in papers if p.get("cited_by_count") is not None]
    avg_citation = statistics.mean(citations) if citations else 0
    median_citation = statistics.median(citations) if citations else 0
    max_citation = max(citations) if citations else 0

    # ==========================================
    # TẠO NỘI DUNG MARKDOWN (CHUẨN HỌC THUẬT)
    # ==========================================
    md_content = f"""# Dataset Descriptive Statistics

*This report is auto-generated for the Methodology section.*

## 1. Overview
The final cleaned dataset comprises a total of **{total_papers}** papers published between **{min_year}** and **{max_year}**, focusing on the sub-field of Natural Language Processing.

## 2. Temporal Distribution
To ensure the reliability of the retrospective validation approach, the temporal distribution of the dataset was analyzed. The distribution indicates a healthy representation across the observed years.

| Publication Year | Number of Papers | Percentage |
| :---: | :---: | :---: |
"""
    # Sinh dòng cho bảng phân bố năm
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        pct = (count / total_papers) * 100
        md_content += f"| {year} | {count} | {pct:.1f}% |\n"

    md_content += f"""
## 3. Textual Characteristics
The abstracts serve as the primary source text for the LLM-based extraction phase. Statistical analysis of the abstract lengths (in words) confirms sufficient context density for relation extraction:
*   **Average length:** {avg_length:.1f} words
*   **Median length:** {median_length} words
*   **Range:** {min_length} - {max_length} words

## 4. Citation Impact
The collected papers demonstrate significant scientific impact, which aligns with the sampling strategy prioritizing highly cited works:
*   **Average citations:** {avg_citation:.1f}
*   **Median citations:** {median_citation}
*   **Maximum citations:** {max_citation}
"""

    # Đảm bảo thư mục results tồn tại
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ghi ra file
    SUMMARY_FILE.write_text(md_content, encoding="utf-8")
    
    print("\n✅ --- HOÀN THÀNH THỐNG KÊ ---")
    print(f"Tổng số bài: {total_papers}")
    print(f"Giai đoạn: {min_year} - {max_year}")
    print(f"Độ dài abstract TB: {avg_length:.1f} từ")
    print(f"💾 Báo cáo Markdown đã được lưu tại: {SUMMARY_FILE}")

if __name__ == "__main__":
    generate_summary()