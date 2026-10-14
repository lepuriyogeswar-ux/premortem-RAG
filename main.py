from __future__ import annotations

from fastapi import FastAPI

from app.api import router
from app.ingestion import load_incidents_from_json
from config import SAMPLE_INCIDENTS_PATH

app = FastAPI(
    title="Premortem RAG",
    version="0.2.0",
    description="Local, evidence-linked failure intelligence prototype. Not a safety authority.",
)
app.state.incidents = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
