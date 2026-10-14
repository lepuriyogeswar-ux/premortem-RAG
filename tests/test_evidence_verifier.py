from __future__ import annotations

from app.evidence_verifier import verify_claim
from app.ingestion import load_incidents_from_json
from config import SAMPLE_INCIDENTS_PATH


def test_matching_claim_gets_source_id():
    incident = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)[0]
    result = verify_claim("Lubrication starvation accelerated bearing wear and heat generation.", [incident])
    assert result.support_status == "SUPPORTED"
    assert result.evidence_incident_ids == ["IND-001"]


def test_missing_evidence_is_not_supported():
    result = verify_claim("A software update caused the failure.", [])
    assert result.support_status == "INSUFFICIENT_EVIDENCE"
    assert result.evidence_incident_ids == []


def test_explicit_negation_is_flagged_for_review():
    incident = load_incidents_from_json(
        [
            {
                "incident_id": "X-1",
                "domain": "test",
                "system_type": "motor",
                "title": "Test",
                "description": "Vibration was observed.",
                "documented_root_cause": "Failure was not caused by vibration.",
                "source_title": "Test source",
                "verification_status": "DOCUMENTED",
            }
        ]
    )[0]
    result = verify_claim("Vibration caused failure.", [incident])
    assert result.support_status in {"CONTRADICTED", "REQUIRES_REVIEW"}
