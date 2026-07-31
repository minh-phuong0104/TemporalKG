import json
import argparse
import os
import time
import signal
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
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "w3leee/CodeX GPT 5.5")

# Endpoint trung gian (proxy) thay vì api.openai.com mặc định.
# Có thể override bằng biến môi trường OPENAI_BASE_URL hoặc --base-url.
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.xah.io/v1")

# Sau mỗi bao nhiêu bài thì ghi checkpoint xuống đĩa.
# Để 1 nghĩa là ghi sau MỖI bài (an toàn nhất, chậm hơn chút do I/O).
DEFAULT_CHECKPOINT_EVERY = 20

# File checkpoint mặc định = <output>.checkpoint.json (nằm cạnh output cuối cùng)
def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".checkpoint.json")


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


def get_openai_client(base_url: str):
    """Khởi tạo client OpenAI, trỏ tới base_url (mặc định là proxy xah.io).
    Cần biến môi trường OPENAI_API_KEY."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Hãy set biến môi trường trước khi chạy, "
            "ví dụ: export OPENAI_API_KEY='...'"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def save_json(path: Path, data) -> None:
    """Ghi an toàn: ghi ra file tạm rồi rename, tránh file bị hỏng
    nếu process bị kill đúng lúc đang ghi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_checkpoint(checkpoint_path: Path) -> list:
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARNING] Không đọc được checkpoint cũ ({e}), bỏ qua và bắt đầu lại từ đầu.")
    return []


def extract_from_abstract(abstract: str, year: int, prompt_template: str, model: str, client,
                           max_retries: int = 3, debug: bool = False, max_tokens: int = 2000) -> list:
    prompt = prompt_template.format(year=year, abstract=abstract)

    if debug:
        print(f"  [DEBUG] abstract length: {len(abstract or '')} chars")
        print(f"  [DEBUG] ===== FULL PROMPT START =====")
        print(prompt)
        print(f"  [DEBUG] ===== FULL PROMPT END =====")

    last_error = None
    current_max_tokens = max_tokens

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,        # deterministic output, giảm drift
                max_tokens=current_max_tokens,
            )
            choice = response.choices[0]
            raw_text = choice.message.content
            finish_reason = choice.finish_reason

            if debug:
                print(f"  [DEBUG] finish_reason: {finish_reason} | max_tokens used: {current_max_tokens}")
                print(f"  [DEBUG] raw_text repr: {raw_text!r}")

            # Trường hợp điển hình của reasoning model (vd Grok qua proxy này):
            # model "nghĩ" trong reasoning_content và ăn hết max_tokens trước khi
            # kịp sinh ra JSON trả lời -> content = None, finish_reason = "length".
            # Cách xử lý: tăng max_tokens lên và thử lại, thay vì bỏ qua bài.
            if not raw_text:
                if finish_reason == "length":
                    reasoning_len = 0
                    try:
                        reasoning_len = len(choice.message.reasoning_content or "")
                    except AttributeError:
                        pass
                    print(f"  [WARNING] Bị cắt giữa chừng do reasoning ăn hết token "
                          f"(reasoning_content ~{reasoning_len} ký tự). Tăng max_tokens và thử lại...")
                    current_max_tokens = min(current_max_tokens * 2, 8000)
                    continue
                print("  [WARNING] Model trả về content rỗng (None/'') không rõ lý do.")
                if debug:
                    print(f"  [DEBUG] full response object: {response}")
                return []

            cleaned = clean_json_text(raw_text)

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, list):
                    print("  [WARNING] Model did not return a JSON list, skipping.")
                    return []
                if debug and len(parsed) == 0:
                    print("  [DEBUG] Model trả về list rỗng '[]' một cách hợp lệ (không tìm thấy quan hệ nào).")
                return parsed
            except json.JSONDecodeError:
                print(f"  [WARNING] Invalid JSON, raw output was:\n  {raw_text[:300]}")
                return []

        except Exception as e:
            last_error = e
            # Hết tiền / hết quota thì retry cũng vô ích -> dừng ngay, không thử lại
            err_str = str(e).lower()
            if "insufficient_quota" in err_str or "insufficient quota" in err_str:
                print(f"  [FATAL] Hết quota API: {e}")
                raise
            wait = min(2 ** attempt, 30)
            print(f"  [WARNING] Lỗi gọi API (lần {attempt}/{max_retries}): {e}. Thử lại sau {wait}s...")
            time.sleep(wait)

    print(f"  [ERROR] Bỏ qua bài này sau {max_retries} lần thử. Lỗi cuối: {last_error}")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract temporal KG quadruples with OpenAI GPT-5.5 (có checkpoint/resume).")
    parser.add_argument("--input", type=Path, default=CLEANED_PAPERS_FILE)
    parser.add_argument("--output", type=Path, default=QUADRUPLES_FILE)
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="Đường dẫn file checkpoint. Mặc định: <output>.checkpoint.json")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                         help="Endpoint API (mặc định: proxy xah.io). Đổi lại nếu muốn dùng thẳng api.openai.com.")
    parser.add_argument("--limit", type=int, default=DEFAULT_EXTRACT_LIMIT, help="Max papers to extract. Use 0 for all.")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                         help="Ghi checkpoint sau mỗi N bài (mặc định 20). Đặt 1 để ghi sau mỗi bài.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Bỏ qua checkpoint cũ, chạy lại từ đầu (mặc định sẽ tự resume nếu có checkpoint).")
    parser.add_argument("--debug", action="store_true",
                         help="In chi tiết abstract, prompt, raw response để debug vì sao ra 0 quadruples.")
    parser.add_argument("--max-tokens", type=int, default=2000,
                         help="Ngân sách token cho completion (mặc định 2000). "
                              "Reasoning model như model đứng sau proxy này 'suy nghĩ' trong cùng ngân sách này, "
                              "nên cần để cao hơn nhiều so với model thường.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or default_checkpoint_path(args.output)

    with open(args.input, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if args.limit and args.limit > 0:
        papers = papers[:args.limit]

    prompt_template = load_prompt_template(args.prompt)
    print(f"[INFO] Dùng endpoint: {args.base_url} | model: {args.model}")
    client = get_openai_client(args.base_url)

    # --- Resume: nạp checkpoint cũ (nếu có) và xác định các paper đã xử lý rồi ---
    all_quadruples = [] if args.no_resume else load_checkpoint(checkpoint_path)
    done_paper_ids = {q.get("paper_id") for q in all_quadruples}
    if all_quadruples:
        print(f"[RESUME] Tìm thấy checkpoint với {len(all_quadruples)} quadruples "
              f"từ {len(done_paper_ids)} bài đã xử lý. Sẽ bỏ qua các bài này.")

    # --- Ghi checkpoint ngay khi bị ngắt (Ctrl+C / SIGTERM) để không mất dữ liệu ---
    def handle_interrupt(signum, frame):
        print(f"\n[INTERRUPTED] Nhận tín hiệu {signum}, đang lưu checkpoint trước khi thoát...")
        save_json(checkpoint_path, all_quadruples)
        print(f"Đã lưu {len(all_quadruples)} quadruples vào {checkpoint_path}")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    processed_count = 0
    try:
        for i, paper in enumerate(papers, 1):
            paper_id = paper.get("id", paper["title"])
            if paper_id in done_paper_ids:
                continue  # đã xử lý ở lần chạy trước, bỏ qua

            print("=" * 60)
            print(f"[{i}/{len(papers)}] {paper['title']}")
            print("=" * 60)

            try:
                quads = extract_from_abstract(paper["abstract"], paper["year"], prompt_template, args.model, client,
                                               debug=args.debug, max_tokens=args.max_tokens)
            except Exception as e:
                # Lỗi nghiêm trọng (vd hết quota): lưu checkpoint rồi dừng hẳn, không chạy tiếp
                print(f"[FATAL] Dừng chạy do lỗi: {e}")
                save_json(checkpoint_path, all_quadruples)
                print(f"Đã lưu checkpoint {len(all_quadruples)} quadruples vào {checkpoint_path} trước khi thoát.")
                raise

            for q in quads:
                q["paper_id"] = paper_id  # traceability: link về bài gốc

            for q in quads:
                print(f"  {q['subject']} --[{q['relation']}]--> {q['object']} ({q['year']})")

            all_quadruples.extend(quads)
            done_paper_ids.add(paper_id)
            processed_count += 1

            if args.checkpoint_every and processed_count % args.checkpoint_every == 0:
                save_json(checkpoint_path, all_quadruples)
                print(f"  [CHECKPOINT] Đã lưu {len(all_quadruples)} quadruples vào {checkpoint_path}")

    finally:
        # Luôn lưu checkpoint lần cuối, dù thành công hay bị lỗi giữa chừng
        save_json(checkpoint_path, all_quadruples)

    # Chạy xong toàn bộ -> ghi ra file output chính thức
    save_json(args.output, all_quadruples)

    print(f"\nExtracted {len(all_quadruples)} quadruples from {len(papers)} papers.")
    print("Saved to", args.output)
    print("Checkpoint file:", checkpoint_path)


if __name__ == "__main__":
    main()