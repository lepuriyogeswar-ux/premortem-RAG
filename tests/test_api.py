from __future__ import annotations

from fastapi.testclient import TestClient

import main
from app.ingestion import load_incidents_from_json
from config import SAMPLE_INCIDENTS_PATH


def test_health_reports_ollama_dependency(monkeypatch):
    monkeypatch.setattr("app.api.get_available_models", lambda: ["qwen3:4b"])
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json()["dependencies"]["ollama"]["reachable"] is True


def test_search_returns_evidence_with_provenance():
    response = TestClient(main.app).post(
        "/search",
        json={"query": "motor overheating vibration", "top_k": 2},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["incident_id"] == "IND-001"
    assert response.json()["results"][0]["source_title"]


def test_empty_search_query_is_rejected():
    response = TestClient(main.app).post("/search", json={"query": " "})
    assert response.status_code == 422


def test_get_incident_not_found_returns_404():
    response = TestClient(main.app).get("/incidents/not-a-real-id")
    assert response.status_code == 404


def test_analyze_does_not_require_ollama():
    response = TestClient(main.app).post(
        "/analyze",
        json={"query": "motor overheating and vibration", "use_llm": False},
    )
    assert response.status_code == 200
    assert response.json()["evidence"]
    assert response.json()["decision"]["decision"] == "INSUFFICIENT_EVIDENCE"
