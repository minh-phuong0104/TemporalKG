import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

RAW_PAPERS_FILE = RAW_DIR / "openalex_2000.json"
RAW_PAPERS_FILE_S2 = RAW_DIR / "semanticscholar_papers.json"
CLEANED_PAPERS_FILE = CLEANED_DIR / "papers_clean.json"

QUADRUPLES_FILE = OUTPUTS_DIR / "quadruples.json"
CLEAN_QUADRUPLES_FILE = OUTPUTS_DIR / "quadruples_clean.json"
REJECTED_QUADRUPLES_FILE = OUTPUTS_DIR / "quadruples_rejected.json"

GRAPH_JSON_FILE = OUTPUTS_DIR / "temporal_kg.json"
GRAPH_GRAPHML_FILE = OUTPUTS_DIR / "temporal_kg.graphml"

DEFAULT_PROMPT_FILE = PROMPTS_DIR / "extraction_v2.txt"

OPENALEX_EMAIL = "daominhhpd74@gmail.com"
SEARCH_CONCEPT = "natural language processing"
YEAR_FROM = 2018
YEAR_TO = 2024
SAMPLE_SIZE = 2000

DEFAULT_EXTRACT_LIMIT = 0  # 0 means extract all cleaned papers

# --- Local model (Ollama) — vẫn giữ để dùng khi cần chạy offline/không tốn API ---
OLLAMA_MODEL = "gemma3:latest"

# --- GPT-5.5 API (extract_triples_gpt.py) ---
# Đọc từ biến môi trường, KHÔNG hardcode key trong file này.
# export GPT_MODEL="gpt-5.5"          (tùy chọn, mặc định đã là gpt-5.5)
# export OPENAI_API_KEY="sk-..."      (bắt buộc để gọi API)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- API keys cào dữ liệu ---
# Tất cả đều đọc từ biến môi trường — KHÔNG hardcode key trực tiếp vào code.
# export OPENALEX_API_KEY="..."
# export SEMANTIC_SCHOLAR_API_KEY="..."
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")