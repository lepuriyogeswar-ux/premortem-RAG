from __future__ import annotations

from app.ingestion import load_incidents_from_json
from app.analysis_service import analyze_problem
from app.risk_engine import assess_evidence
from config import SAMPLE_INCIDENTS_PATH


def test_empty_evidence_is_insufficient():
    result = assess_evidence([])
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["uncertainty"] == "HIGH"


def test_safety_critical_always_requires_expert_review():
    evidence = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    result = assess_evidence(evidence, safety_critical=True)
    assert result["decision"] == "REQUIRES_EXPERT_REVIEW"
    assert "Do not bypass" in result["recommended_next_action"]


def test_synthetic_sources_are_not_strong_real_world_evidence():
    evidence = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    result = assess_evidence(evidence)
    assert result["evidence_strength"] == "DEMONSTRATION_ONLY"
    assert result["decision"] == "INSUFFICIENT_EVIDENCE"


def test_synthetic_claims_are_never_presented_as_verified_facts():
    evidence = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    result = analyze_problem("motor overheating and vibration", evidence)
    assert result["claims"]
    assert all(claim.support_status == "REQUIRES_REVIEW" for claim in result["claims"])
