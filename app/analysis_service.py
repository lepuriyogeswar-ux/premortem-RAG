from __future__ import annotations

from collections.abc import Sequence

from app.evidence_verifier import verify_claim
from app.llm_service import OllamaError, generate_analysis
from app.retrieval import search_incidents
from app.risk_engine import assess_evidence
from app.schemas import IncidentRecord


def analyze_problem(
    query: str,
    incidents: Sequence[IncidentRecord],
    *,
    domain: str | None = None,
    system_type: str | None = None,
    top_k: int = 5,
    model: str = "qwen3:4b",
    use_llm: bool = False,
    safety_critical: bool = False,
) -> dict:
    evidence = search_incidents(query, incidents, top_k, domain, system_type)
    by_id = {incident.incident_id: incident for incident in incidents}
    retrieved = [by_id[result.incident_id] for result in evidence]
    claims = []
    chains = []
    for incident in retrieved:
        if incident.documented_root_cause:
            claim = verify_claim(incident.documented_root_cause, [incident])
            if claim.support_status == "SUPPORTED" and (
                incident.is_synthetic or incident.verification_status != "DOCUMENTED"
            ):
                claim.support_status = "REQUIRES_REVIEW"
                claim.explanation = (
                    "Text overlap was found, but this record is synthetic or not source-verified; "
                    "it cannot establish the claim as a real incident fact."
                )
            claim.source_passage = incident.relevant_passage or incident.documented_root_cause
            claim.source_reference = incident.source_url or incident.source_file or incident.source_title
            claims.append(claim)
        for relation, values in (
            ("Observation", incident.observed_symptoms),
            ("Contributing Condition", incident.contributing_conditions),
            ("Failure Mechanism", [incident.failure_mechanism] if incident.failure_mechanism else []),
            ("Potential Outcome", [incident.outcome_or_impact] if incident.outcome_or_impact else []),
            ("Diagnostic / Preventive Check", incident.preventive_actions),
        ):
            for value in values:
                chains.append(
                    {
                        "incident_id": incident.incident_id,
                        "source_reference": incident.source_url or incident.source_file or incident.source_title,
                        "element": relation,
                        "text": value,
                        "evidence_status": (
                            "DOCUMENTED"
                            if incident.verification_status == "DOCUMENTED" and not incident.is_synthetic
                            else "INFERRED"
                            if incident.verification_status == "DOCUMENTED"
                            else "UNKNOWN"
                        ),
                        "is_synthetic": incident.is_synthetic,
                    }
                )

    generated_analysis = None
    model_warning = None
    if use_llm:
        try:
            generated_analysis = generate_analysis(
                query,
                model,
                [
                    {
                        "incident_id": item.incident_id,
                        "domain": item.domain,
                        "title": item.title,
                        "passage": item.relevant_passage or item.description,
                        "source": item.source_url or item.source_file or item.source_title,
                        "synthetic": item.is_synthetic,
                    }
                    for item in retrieved
                ],
            )
        except OllamaError as exc:
            model_warning = str(exc)

    return {
        "query": query,
        "evidence": evidence,
        "claims": claims,
        "failure_chains": chains,
        "decision": assess_evidence(retrieved, safety_critical=safety_critical),
        "generated_analysis": generated_analysis,
        "model_warning": model_warning,
    }
