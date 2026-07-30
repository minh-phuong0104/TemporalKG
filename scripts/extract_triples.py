import json
import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    CLEANED_PAPERS_FILE,
    DEFAULT_EXTRACT_LIMIT,
    DEFAULT_PROMPT_FILE,
    OLLAMA_MODEL,
    QUADRUPLES_FILE,
)


def clean_json_text(raw: str) -> str:
    """Defensive cleanup: smaller local models don't always follow
    'no markdown' instructions as reliably as larger hosted models."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return text


def load_prompt_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def extract_from_abstract(abstract: str, year: int, prompt_template: str, model: str) -> list:
    from ollama import chat

    prompt = prompt_template.format(year=year, abstract=abstract)

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,     # deterministic output, reduces drift
            "num_predict": 800,   # avoid truncating mid-JSON on longer outputs
        },
    )
    raw_text = response["message"]["content"]
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
    parser = argparse.ArgumentParser(description="Extract temporal KG quadruples with Ollama.")
    parser.add_argument("--input", type=Path, default=CLEANED_PAPERS_FILE)
    parser.add_argument("--output", type=Path, default=QUADRUPLES_FILE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_EXTRACT_LIMIT, help="Max papers to extract. Use 0 for all.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if args.limit and args.limit > 0:
        papers = papers[:args.limit]

    prompt_template = load_prompt_template(args.prompt)
    all_quadruples = []

    for i, paper in enumerate(papers, 1):
        print("=" * 60)
        print(f"[{i}/{len(papers)}] {paper['title']}")
        print("=" * 60)

        quads = extract_from_abstract(paper["abstract"], paper["year"], prompt_template, args.model)

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
