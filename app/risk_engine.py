from __future__ import annotations

from collections.abc import Sequence

from app.schemas import IncidentRecord


def assess_evidence(
    evidence: Sequence[IncidentRecord],
    *,
    safety_critical: bool = False,
) -> dict[str, str]:
    """Return transparent decision labels; scores are not probabilities."""
    if safety_critical:
        return {
            "evidence_strength": "LOW" if not evidence else "LIMITED",
            "potential_severity": "HIGH",
            "uncertainty": "HIGH" if len(evidence) < 2 else "MODERATE",
            "decision": "REQUIRES_EXPERT_REVIEW",
            "recommended_next_action": "Follow approved procedures and consult qualified safety/engineering personnel. Do not bypass safeguards.",
        }
    if not evidence:
        return {
            "evidence_strength": "NONE",
            "potential_severity": "UNKNOWN",
            "uncertainty": "HIGH",
            "decision": "INSUFFICIENT_EVIDENCE",
            "recommended_next_action": "Gather system details, operating conditions, and traceable incident sources before drawing conclusions.",
        }

    documented = sum(item.verification_status == "DOCUMENTED" for item in evidence)
    synthetic = sum(item.is_synthetic for item in evidence)
    if documented >= 2 and synthetic == 0:
        strength = "MODERATE"
        decision = "INVESTIGATE"
    elif documented >= 1 and synthetic == 0:
        strength = "LIMITED"
        decision = "INVESTIGATE"
    else:
        strength = "DEMONSTRATION_ONLY"
        decision = "INSUFFICIENT_EVIDENCE"

    return {
        "evidence_strength": strength,
        "potential_severity": "UNKNOWN",
        "uncertainty": "HIGH" if len(evidence) < 3 or synthetic else "MODERATE",
        "decision": decision,
        "recommended_next_action": "Use the retrieved records to prioritize inspection questions; confirm findings against approved procedures and primary sources.",
    }
