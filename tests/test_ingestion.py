from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion import deduplicate_records, load_incidents_from_json, validate_incident_record


VALID_RECORD = {
    "incident_id": "IND-100",
    "domain": "industrial systems",
    "system_type": "motor",
    "title": "Vibration increase on compressor",
    "description": "Unplanned vibration rise preceded bearing wear.",
    "observed_symptoms": ["High vibration", "Elevated temperature"],
    "contributing_conditions": ["Misalignment"],
    "documented_root_cause": "Shaft misalignment caused bearing stress.",
    "failure_mechanism": "Dynamic imbalance created heat and wear.",
    "outcome_or_impact": "Production stopped for inspection.",
    "corrective_actions": ["Align shaft"],
    "preventive_actions": ["Vibration monitoring"],
    "source_title": "Compressor case summary",
    "source_url": "https://example.com/compressor-case",
    "incident_date": "2024-01-15",
    "provenance": "sample",
    "verification_status": "DOCUMENTED",
    "is_synthetic": True,
    "relevant_passage": "Vibration rose before the failure."
}


def test_validate_incident_allows_valid_record():
    record = validate_incident_record(VALID_RECORD)
    assert record.incident_id == "IND-100"
    assert record.summary_text


def test_invalid_record_rejected():
    with pytest.raises(Exception):
        validate_incident_record({
            "incident_id": "",
            "domain": "",
            "system_type": "motor",
            "title": "bad record",
            "description": "bad description",
            "observed_symptoms": ["bad"],
            "source_title": "source"
        })


def test_deduplicate_records_removes_duplicates():
    records = [VALID_RECORD, {**VALID_RECORD, "incident_id": "IND-100"}]
    deduped, removed = deduplicate_records(records)
    assert len(deduped) == 1
    assert removed == 1


def test_load_incidents_from_json_file(tmp_path):
    path = tmp_path / "incidents.json"
    path.write_text(json.dumps([VALID_RECORD]), encoding="utf-8")
    records = load_incidents_from_json(path)
    assert len(records) == 1
    assert records[0].title == VALID_RECORD["title"]


def test_sample_incidents_file_exists():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_incidents.json"
    assert sample.exists()
    payload = json.loads(sample.read_text(encoding="utf-8"))
    assert len(payload) >= 2
