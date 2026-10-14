from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.ingestion import load_incidents_from_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest incident records into the local Premortem RAG dataset")
    parser.add_argument("path", type=str, help="Path to a JSON or CSV incident file.")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    try:
        records = load_incidents_from_json(path) if path.suffix.lower() == ".json" else __import__("app.ingestion", fromlist=["load_incidents_from_csv"]).load_incidents_from_csv(path)
    except Exception as exc:  # pragma: no cover - CLI path only
        print(f"Failed to ingest: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} incidents from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
