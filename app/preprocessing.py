from __future__ import annotations

"""Preprocessing utilities for failing incidents and text normalization."""


def normalize_incident_text(text: str) -> str:
    return " ".join(text.strip().split())
