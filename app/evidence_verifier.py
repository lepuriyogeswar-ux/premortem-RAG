from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas import EvidenceClaim, IncidentRecord

_WORDS = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def _terms(text: str) -> set[str]:
    return {word for word in _WORDS.findall(text.lower()) if word not in _STOPWORDS}


def verify_claim(claim: str, evidence: Sequence[IncidentRecord]) -> EvidenceClaim:
    claim_terms = _terms(claim)
    if not claim_terms or not evidence:
        return EvidenceClaim(
            claim=claim,
            evidence_incident_ids=[],
            support_status="INSUFFICIENT_EVIDENCE",
            explanation="No usable claim terms or retrieved incident evidence were provided.",
        )

    supported_ids: list[str] = []
    opposing_ids: list[str] = []
    for incident in evidence:
        factual_text = " ".join(
            filter(
                None,
                [
                    incident.relevant_passage,
                    incident.description,
                    incident.documented_root_cause,
                    incident.failure_mechanism,
                    incident.outcome_or_impact,
                    *incident.observed_symptoms,
                    *incident.contributing_conditions,
                ],
            )
        )
        overlap = claim_terms & _terms(factual_text)
        coverage = len(overlap) / len(claim_terms)
        if coverage >= 0.5 and len(overlap) >= min(2, len(claim_terms)):
            supported_ids.append(incident.incident_id)

        cause = incident.documented_root_cause or ""
        # Lexically explicit negation is treated as a conflict, not as a general NLI claim.
        if any(marker in cause.casefold() for marker in ("not caused by", "was not due to", "ruled out")):
            if _terms(cause) & claim_terms:
                opposing_ids.append(incident.incident_id)

    if supported_ids and opposing_ids:
        status = "REQUIRES_REVIEW"
        explanation = "Retrieved passages contain lexical support and an explicit conflicting statement; review both sources."
        ids = sorted(set(supported_ids + opposing_ids))
    elif opposing_ids:
        status = "CONTRADICTED"
        explanation = "A retrieved source explicitly negates this cause; this lexical check requires human review."
        ids = opposing_ids
    elif supported_ids:
        status = "SUPPORTED"
        explanation = "The claim shares key terms with retrieved source passages. This is a lexical check, not proof of causation."
        ids = supported_ids
    else:
        status = "INSUFFICIENT_EVIDENCE"
        explanation = "The retrieved passages do not contain enough overlapping terms to support the claim."
        ids = []

    return EvidenceClaim(
        claim=claim,
        evidence_incident_ids=ids,
        support_status=status,
        explanation=explanation,
    )
