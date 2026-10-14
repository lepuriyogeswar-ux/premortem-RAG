from __future__ import annotations

from app.ingestion import load_incidents_from_json
from app.retrieval import search_incidents
from config import SAMPLE_INCIDENTS_PATH


def test_hybrid_retrieval_ranks_motor_incident_for_motor_symptoms():
    incidents = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    result = search_incidents("motor overheating increased vibration and power draw", incidents)
    assert result
    assert result[0].incident_id == "IND-001"
    assert result[0].source_title
    assert result[0].relevant_passage


def test_vector_baseline_returns_provenance():
    incidents = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    result = search_incidents("pump cavitation low suction head", incidents, method="vector")
    assert result[0].incident_id == "IND-002"
    assert result[0].source_url == "https://example.com/pump-cavitation-review"


def test_search_rejects_blank_query_and_respects_domain_filter():
    incidents = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    try:
        search_incidents(" ", incidents)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Blank query was not rejected")
    result = search_incidents("motor vibration", incidents, domain="pipe and fluid systems")
    assert all(item.domain == "pipe and fluid systems" for item in result)
