import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

RAW_PAPERS_FILE = RAW_DIR / "openalex_2000.json"
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

OLLAMA_MODEL = "gemma3:latest"
DEFAULT_EXTRACT_LIMIT = 0  # 0 means extract all cleaned papers
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
