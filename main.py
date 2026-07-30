"""
Run the TemporalKG pipeline with one command.

Default order:
collect -> clean -> extract -> filter -> build_graph

Examples:
    python main.py --skip-collect
    python main.py --skip-collect --extract-limit 10
    python main.py --collect-limit 2000
"""

import argparse
import subprocess
import sys


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full TemporalKG pipeline.")
    parser.add_argument("--skip-collect", action="store_true", help="Use existing data/raw/openalex_2000.json.")
    parser.add_argument("--collect-limit", type=int, default=None, help="Override OpenAlex collection size.")
    parser.add_argument("--extract-limit", type=int, default=None, help="Limit papers for extraction. Use 0 for all.")
    parser.add_argument("--model", default=None, help="Override Ollama model.")
    parser.add_argument("--prompt", default=None, help="Override prompt file.")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_collect:
        collect_cmd = [python, "-m", "scripts.collect_openalex"]
        if args.collect_limit is not None:
            collect_cmd += ["--limit", str(args.collect_limit)]
        run_step("collect", collect_cmd)

    run_step("clean", [python, "-m", "scripts.clean_data"])

    extract_cmd = [python, "-m", "scripts.extract_triples"]
    if args.extract_limit is not None:
        extract_cmd += ["--limit", str(args.extract_limit)]
    if args.model:
        extract_cmd += ["--model", args.model]
    if args.prompt:
        extract_cmd += ["--prompt", args.prompt]
    run_step("extract", extract_cmd)

    run_step("filter", [python, "-m", "scripts.post_filter"])
    run_step("build_graph", [python, "-m", "scripts.build_graph"])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
