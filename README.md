# Premortem RAG

A local-first failure-intelligence prototype that retrieves documented incident records, preserves their source references, and distinguishes evidence from hypotheses. It is decision support, not an autonomous safety authority.

## Run the browser dashboard

From PowerShell in a clone of this repository:

```powershell
cd "d:\premortem RAG\premortem-rag"
python -m streamlit run ".\frontend\streamlit_app.py"
```

Streamlit prints the local URL (normally http://localhost:8501) and opens the dashboard in a browser. From the original workspace parent folder, `python -m streamlit run ".\premortem-rag\frontend\streamlit_app.py"` also works.

## Run the API

```powershell
cd "d:\premortem RAG\premortem-rag"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Visit http://127.0.0.1:8000/docs for interactive API documentation.

## Environment and dependencies

- Python 3.10+; Python 3.13.9 was available during implementation.
- Install project dependencies if needed:

  ```powershell
  cd "d:\premortem RAG\premortem-rag"
  python -m pip install -r requirements.txt
  ```

- Ollama is optional for search and the deterministic analysis. The optional narrative uses Ollama at `http://localhost:11434`, defaulting to `qwen3:4b`; `llama3.2:3b` is also selectable. No data is sent to hosted LLM APIs.

## Current retrieval implementation

The runnable prototype uses local TF-IDF vectors as a no-download vector baseline, an in-project BM25 implementation, and reciprocal-rank fusion. It does **not** yet use pretrained Sentence Transformers embeddings or Qdrant. It reports source passages and provenance metadata; retrieval similarity does not prove causation.

The dataset in `data/sample_incidents.json` is synthetic demonstration content and must not be represented as real incident evidence. JSON and CSV uploads are validated in the dashboard; uploaded records remain in the current Streamlit session.

## API

- `GET /health` — app and Ollama status (degraded if Ollama cannot be reached)
- `GET /models` — Ollama models available locally
- `POST /incidents/ingest` — validate and append records to this process's in-memory dataset
- `POST /search` — hybrid retrieval
- `POST /analyze` — evidence-linked result and optional local Ollama narrative
- `GET /incidents/{incident_id}` — retrieve one incident

The API dataset is in memory and resets when its process restarts. Uploaded UI records are separate from the API process.

## Tests

```powershell
cd "d:\premortem RAG\premortem-rag"
python -m pytest -q
```

## Limitations

- TF-IDF is lexical vectorization, not a pretrained semantic embedding model.
- The claim verifier is a conservative lexical check, not a validated natural-language inference system. Its labels need human review.
- The risk engine exposes qualitative evidence/uncertainty labels, not calibrated probabilities or validated risk scores.
- Failure-chain entries are structured source-linked lists, not causal proof or a validated causal graph.
- Synthetic sample records cannot establish that a real-world failure mechanism is likely.
- Safety-critical decisions require approved procedures and qualified personnel.
