from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.analysis_service import analyze_problem
from app.ingestion import validate_and_deduplicate_records
from app.llm_service import ALLOWED_MODELS, OllamaError, get_available_models
from app.retrieval import search_incidents
from app.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    IncidentIngestRequest,
    IncidentRecord,
    SearchQueryRequest,
    SearchResponse,
)

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    try:
        available_models = get_available_models()
        ollama = {"reachable": True, "models": available_models}
    except OllamaError as exc:
        ollama = {"reachable": False, "models": [], "message": str(exc)}
    store: list[IncidentRecord] = request.app.state.incidents
    return {
        "status": "ok" if ollama["reachable"] else "degraded",
        "service": "premortem-rag",
        "incident_count": len(store),
        "dependencies": {"ollama": ollama},
    }


@router.get("/models")
def models() -> dict:
    try:
        available = get_available_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"models": available, "supported": sorted(ALLOWED_MODELS)}


@router.post("/incidents/ingest")
def ingest(payload: IncidentIngestRequest, request: Request) -> JSONResponse:
    if len(payload.records) > 5000:
        raise HTTPException(status_code=413, detail="At most 5000 incident records may be submitted at once.")
    records, result = validate_and_deduplicate_records(payload.records)
    store: list[IncidentRecord] = request.app.state.incidents
    existing = {item.incident_id.casefold() for item in store}
    new_records = [item for item in records if item.incident_id.casefold() not in existing]
    result.duplicates_removed += len(records) - len(new_records)
    store.extend(new_records)
    result.ingested = len(new_records)
    result.incident_ids = [item.incident_id for item in new_records]
    status_code = 207 if result.invalid_records else 200
    return JSONResponse(status_code=status_code, content=result.model_dump())


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchQueryRequest, request: Request) -> SearchResponse:
    try:
        results = search_incidents(
            payload.query,
            request.app.state.incidents,
            payload.top_k,
            payload.domain,
            payload.system_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SearchResponse(query=payload.query, results=results)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: AnalyzeRequest, request: Request) -> dict:
    return analyze_problem(
        payload.query,
        request.app.state.incidents,
        domain=payload.domain,
        system_type=payload.system_type,
        top_k=payload.top_k,
        model=payload.model,
        use_llm=payload.use_llm,
        safety_critical=payload.safety_critical,
    )


@router.get("/incidents/{incident_id}", response_model=IncidentRecord)
def get_incident(incident_id: str, request: Request) -> IncidentRecord:
    for item in request.app.state.incidents:
        if item.incident_id.casefold() == incident_id.casefold():
            return item
    raise HTTPException(status_code=404, detail="Incident not found.")
