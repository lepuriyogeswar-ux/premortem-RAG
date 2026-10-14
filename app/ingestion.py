from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from app.schemas import IncidentRecord, IngestResult


def validate_incident_record(payload: dict[str, Any]) -> IncidentRecord:
    if not isinstance(payload, dict):
        raise ValueError("Incident record must be a JSON object.")
    return IncidentRecord.model_validate(payload)


def deduplicate_records(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0
    for record in records:
        incident_id = record.get("incident_id")
        if not isinstance(incident_id, str) or not incident_id.strip():
            deduped.append(record)
            continue
        key = incident_id.strip().casefold()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(record)
    return deduped, removed


def load_incidents_from_json(source: Any) -> list[IncidentRecord]:
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, list):
        data = source
    elif isinstance(source, dict):
        data = [source]
    else:
        raise TypeError("JSON input must be a dict, list of dicts, or a file path.")
    if not isinstance(data, list):
        raise ValueError("The JSON payload must contain a list of incident records.")
    return [validate_incident_record(item) for item in data]


def _parse_csv_list(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(";") if part.strip()]
    return decoded if isinstance(decoded, list) else [value]


def load_incidents_from_csv(source: str | Path | io.TextIOBase) -> list[IncidentRecord]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8-sig", newline="") as handle:
            return load_incidents_from_csv(handle)
    reader = csv.DictReader(source)
    records: list[IncidentRecord] = []
    list_fields = {"observed_symptoms", "contributing_conditions", "corrective_actions", "preventive_actions"}
    for row in reader:
        normalized = dict(row)
        for field in list_fields:
            if field in normalized:
                normalized[field] = _parse_csv_list(normalized[field])
        for field in ("documented_root_cause", "failure_mechanism", "outcome_or_impact", "source_url", "source_file", "incident_date", "relevant_passage"):
            if normalized.get(field) == "":
                normalized[field] = None
        if isinstance(normalized.get("is_synthetic"), str):
            normalized["is_synthetic"] = normalized["is_synthetic"].strip().casefold() in {"true", "1", "yes"}
        records.append(validate_incident_record(normalized))
    return records


def validate_and_deduplicate_records(payload: Any) -> tuple[list[IncidentRecord], IngestResult]:
    if not isinstance(payload, list):
        raise ValueError("The request must contain a list of incident record objects.")
    valid: list[IncidentRecord] = []
    invalid_details: list[str] = []
    duplicates_removed = 0
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        try:
            incident = validate_incident_record(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            invalid_details.append(f"Record {index + 1}: {exc}")
            continue
        key = incident.incident_id.casefold()
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        valid.append(incident)
    result = IngestResult(
        ingested=len(valid),
        duplicates_removed=duplicates_removed,
        invalid_records=len(invalid_details),
        invalid_details=invalid_details,
        incident_ids=[incident.incident_id for incident in valid],
    )
    return valid, result


def iter_incident_chunks(records: Iterable[IncidentRecord]) -> list[str]:
    chunks: list[str] = []
    for item in records:
        text = "\n".join(
            [
                item.title,
                item.description,
                *item.observed_symptoms,
                *item.contributing_conditions,
                item.failure_mechanism or "",
                item.documented_root_cause or "",
                item.relevant_passage or "",
            ]
        )
        chunks.append(text.strip())
    return [chunk for chunk in chunks if chunk]
