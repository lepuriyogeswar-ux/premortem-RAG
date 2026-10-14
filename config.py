from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_INCIDENTS_PATH = DATA_DIR / "sample_incidents.json"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))

# Safety/control defaults for ingestion and UI
MAX_INCIDENTS_PER_UPLOAD = 5000
MAX_TEXT_LENGTH = 20000
MAX_REQUEST_BYTES = 5 * 1024 * 1024
