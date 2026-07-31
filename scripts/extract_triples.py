import json
import argparse
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    CLEANED_PAPERS_FILE,
    DEFAULT_EXTRACT_LIMIT,
    DEFAULT_PROMPT_FILE,
    QUADRUPLES_FILE,
)

# Model GPT-5.5 của OpenAI (thay cho OLLAMA_MODEL trước đây).
# Có thể override bằng biến môi trường OPENAI_MODEL hoặc --model.
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")


def clean_json_text(raw: str) -> str:
    """Defensive cleanup: đôi khi model vẫn bọc kết quả trong code block
    dù đã được yêu cầu 'no markdown'."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return text


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def get_openai_client():
    """Khởi tạo client OpenAI. Cần biến môi trường OPENAI_API_KEY."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Hãy set biến môi trường trước khi chạy, "
            "ví dụ: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def extract_from_abstract(abstract: str, year: int, prompt_template: str, model: str, client) -> list:
    prompt = prompt_template.format(year=year, abstract=abstract)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,        # deterministic output, giảm drift
        max_tokens=800,       # tránh bị cắt giữa JSON với output dài
    )
    raw_text = response.choices[0].message.content
    cleaned = clean_json_text(raw_text)

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            print("  [WARNING] Model did not return a JSON list, skipping.")
            return []
        return parsed
    except json.JSONDecodeError:
        print(f"  [WARNING] Invalid JSON, raw output was:\n  {raw_text[:300]}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract temporal KG quadruples with OpenAI GPT-5.5.")
    parser.add_argument("--input", type=Path, default=CLEANED_PAPERS_FILE)
    parser.add_argument("--output", type=Path, default=QUADRUPLES_FILE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_EXTRACT_LIMIT, help="Max papers to extract. Use 0 for all.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if args.limit and args.limit > 0:
        papers = papers[:args.limit]

    prompt_template = load_prompt_template(args.prompt)
    client = get_openai_client()
    all_quadruples = []

    for i, paper in enumerate(papers, 1):
        print("=" * 60)
        print(f"[{i}/{len(papers)}] {paper['title']}")
        print("=" * 60)

        quads = extract_from_abstract(paper["abstract"], paper["year"], prompt_template, args.model, client)

        # traceability: always keep a link back to the source paper
        for q in quads:
            q["paper_id"] = paper.get("id", paper["title"])

        for q in quads:
            print(f"  {q['subject']} --[{q['relation']}]--> {q['object']} ({q['year']})")

        all_quadruples.extend(quads)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_quadruples, f, ensure_ascii=False, indent=2)

    print(f"\nExtracted {len(all_quadruples)} quadruples from {len(papers)} papers.")
    print("Saved to", args.output)


if __name__ == "__main__":
    main()