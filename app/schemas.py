from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IncidentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(..., min_length=1, max_length=128)
    domain: str = Field(..., min_length=1, max_length=120)
    system_type: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=20000)
    observed_symptoms: List[str] = Field(default_factory=list, max_length=100)
    contributing_conditions: List[str] = Field(default_factory=list, max_length=100)
    documented_root_cause: Optional[str] = Field(default=None, max_length=5000)
    failure_mechanism: Optional[str] = Field(default=None, max_length=5000)
    outcome_or_impact: Optional[str] = Field(default=None, max_length=5000)
    corrective_actions: List[str] = Field(default_factory=list, max_length=100)
    preventive_actions: List[str] = Field(default_factory=list, max_length=100)
    source_title: str = Field(..., min_length=1, max_length=500)
    source_url: Optional[str] = Field(default=None, max_length=2000)
    source_file: Optional[str] = Field(default=None, max_length=500)
    incident_date: Optional[str] = Field(default=None, max_length=40)
    provenance: str = Field(default="unknown", max_length=1000)
    verification_status: Literal["DOCUMENTED", "PARTIAL", "UNVERIFIED"] = "UNVERIFIED"
    is_synthetic: bool = False
    relevant_passage: Optional[str] = Field(default=None, max_length=20000)

    @field_validator("observed_symptoms", "contributing_conditions", "corrective_actions", "preventive_actions")
    @classmethod
    def ensure_list_items_are_strings(cls, value: List[str]) -> List[str]:
        if value is None:
            return []
        cleaned = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, str):
                raise TypeError("List entries must be strings.")
            cleaned.append(item.strip())
        return [item for item in cleaned if item]

    @field_validator("incident_id")
    @classmethod
    def sanitize_incident_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("incident_id cannot be blank.")
        return normalized

    @property
    def summary_text(self) -> str:
        return "\n".join(
            filter(
                None,
                [
                    self.title,
                    self.description,
                    " ".join(self.observed_symptoms),
                    self.failure_mechanism,
                    self.documented_root_cause,
                    self.outcome_or_impact,
                ],
            )
        )


class IncidentIngestRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list, max_length=5000)


class IngestResult(BaseModel):
    ingested: int
    duplicates_removed: int
    invalid_records: int
    invalid_details: List[str] = Field(default_factory=list)
    incident_ids: List[str] = Field(default_factory=list)


class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    domain: Optional[str] = None
    system_type: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultItem(BaseModel):
    incident_id: str
    domain: str
    system_type: str
    title: str
    source_title: str
    source_url: Optional[str] = None
    score: float
    provenance: str
    relevant_passage: str
    same_domain: bool


class SearchResponse(BaseModel):
    results: List[SearchResultItem] = Field(default_factory=list)
    query: str
    warning: Optional[str] = None


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    domain: Optional[str] = None
    system_type: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    model: Literal["qwen3:4b", "llama3.2:3b"] = "qwen3:4b"
    use_llm: bool = False
    safety_critical: bool = False


class EvidenceClaim(BaseModel):
    claim: str
    evidence_incident_ids: List[str] = Field(default_factory=list)
    support_status: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "REQUIRES_REVIEW"]
    explanation: str
    source_passage: Optional[str] = None
    source_reference: Optional[str] = None


class AnalysisResponse(BaseModel):
    query: str
    evidence: List[SearchResultItem] = Field(default_factory=list)
    claims: List[EvidenceClaim] = Field(default_factory=list)
    failure_chains: List[Dict[str, Any]] = Field(default_factory=list)
    decision: Dict[str, str]
    generated_analysis: Optional[str] = None
    model_warning: Optional[str] = None
