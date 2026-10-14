from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.analysis_service import analyze_problem
from app.ingestion import load_incidents_from_csv, load_incidents_from_json, validate_and_deduplicate_records
from app.llm_service import OllamaError, get_available_models
from app.schemas import IncidentRecord
from config import MAX_INCIDENTS_PER_UPLOAD, MAX_REQUEST_BYTES, SAMPLE_INCIDENTS_PATH


def _uploaded_records(uploaded_file) -> tuple[list[IncidentRecord], list[str]]:
    data = uploaded_file.getvalue()
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("Upload exceeds the 5 MB limit.")
    suffix = Path(uploaded_file.name).suffix.casefold()
    if suffix == ".json":
        payload = json.loads(data.decode("utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError("JSON upload must contain a list of incident records.")
        records, result = validate_and_deduplicate_records(payload)
        return records, result.invalid_details
    if suffix == ".csv":
        return load_incidents_from_csv(io.StringIO(data.decode("utf-8-sig"))), []
    raise ValueError("Upload a .json or .csv file.")


def _source_link(url: str | None) -> None:
    if url and url.startswith(("https://", "http://")):
        st.link_button("Open source", url)


def render_app() -> None:
    st.set_page_config(page_title="Premortem RAG", page_icon="🧭", layout="wide")
    st.title("Premortem RAG")
    st.caption("Cross-domain failure intelligence • local-first • evidence-linked decision support")
    st.warning(
        "Decision-support prototype only. Not a substitute for qualified engineering, medical, "
        "pharmaceutical, or safety-critical judgment. Follow approved procedures."
    )

    try:
        records = load_incidents_from_json(SAMPLE_INCIDENTS_PATH)
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"Could not load sample incidents: {exc}")
        st.stop()

    if "extra_incidents" not in st.session_state:
        st.session_state.extra_incidents = []

    with st.sidebar:
        st.header("Investigation settings")
        domain_values = sorted({item.domain for item in records + st.session_state.extra_incidents})
        selected_domain = st.selectbox("Domain filter", ["All domains", *domain_values])
        domain = None if selected_domain == "All domains" else selected_domain

        system_values = sorted(
            {
                item.system_type
                for item in records + st.session_state.extra_incidents
                if domain is None or item.domain.casefold() == domain.casefold()
            }
        )
        selected_system = st.selectbox("System type", ["All systems", *system_values])
        system_type = None if selected_system == "All systems" else selected_system
        top_k = st.slider("Evidence results", min_value=1, max_value=10, value=5)
        model = st.selectbox("Local Ollama model", ["qwen3:4b", "llama3.2:3b"])
        use_llm = st.checkbox("Generate a narrative with Ollama", value=False)
        safety_critical = st.checkbox("Safety-critical system", value=False)

        if st.button("Check Ollama connection"):
            try:
                available = get_available_models()
                st.success("Available locally: " + (", ".join(available) or "no models found"))
            except OllamaError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Add incident records")
        upload = st.file_uploader("JSON or CSV (max 5 MB)", type=["json", "csv"])
        if upload is not None and st.button("Validate and add records"):
            try:
                imported, errors = _uploaded_records(upload)
                if len(imported) + len(st.session_state.extra_incidents) > MAX_INCIDENTS_PER_UPLOAD:
                    st.error("Incident limit exceeded (5000 records).")
                else:
                    known = {item.incident_id.casefold() for item in records + st.session_state.extra_incidents}
                    added = [item for item in imported if item.incident_id.casefold() not in known]
                    st.session_state.extra_incidents.extend(added)
                    st.success(f"Added {len(added)} records; {len(imported) - len(added)} duplicates skipped.")
                    if errors:
                        st.warning(f"{len(errors)} record(s) were invalid.")
                        for error in errors[:10]:
                            st.caption(error)
            except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
                st.error(f"Upload could not be read: {exc}")

    incidents = records + st.session_state.extra_incidents
    st.caption(f"Indexed incidents: {len(incidents)} • sample dataset records are synthetic")
    problem = st.text_area(
        "Describe the problem",
        value="Our industrial motor is overheating, vibration has increased, and power consumption is abnormal. What should we investigate first?",
        max_chars=2000,
        height=110,
        help="Describe observed symptoms and conditions. Do not include secrets or personal information.",
    )
    analyze = st.button("Analyze evidence", type="primary", disabled=not problem.strip())

    if analyze:
        with st.spinner("Retrieving evidence and checking claims..."):
            try:
                result = analyze_problem(
                    problem,
                    incidents,
                    domain=domain,
                    system_type=system_type,
                    top_k=top_k,
                    model=model,
                    use_llm=use_llm,
                    safety_critical=safety_critical,
                )
                st.session_state.last_analysis = result
            except (ValueError, RuntimeError) as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("last_analysis")
    if not result:
        st.info("Enter a problem description and select **Analyze evidence** to begin.")
        with st.expander("What this prototype can and cannot do"):
            st.markdown(
                "- Search sample or uploaded JSON/CSV incident records.\n"
                "- Compare a local TF-IDF vector baseline with BM25 keyword retrieval using reciprocal-rank fusion.\n"
                "- Keep incident IDs, source references, and passages attached to results.\n"
                "- This prototype does not use pretrained semantic embeddings. Similarity is not proof of causation.\n"
                "- Synthetic sample records are for UI demonstration only, not real evidence."
            )
        return

    st.subheader("Decision summary")
    decision = result["decision"]
    cols = st.columns(4)
    for col, label in zip(
        cols,
        ("evidence_strength", "potential_severity", "uncertainty", "decision"),
    ):
        col.metric(label.replace("_", " ").title(), decision[label])
    st.write(decision["recommended_next_action"])

    st.subheader("Retrieved evidence")
    if not result["evidence"]:
        st.info("No incident matched the selected filters. Broaden the filters or add more records.")
    for item in result["evidence"]:
        with st.container(border=True):
            header = f"{item.incident_id} · {item.title}"
            st.markdown(f"**{header}**")
            st.caption(
                f"{item.domain} / {item.system_type} · score {item.score:.4f} · "
                f"{item.provenance or 'provenance unknown'}"
            )
            if next(
                (incident.is_synthetic for incident in incidents if incident.incident_id == item.incident_id),
                False,
            ):
                st.warning("Synthetic demonstration record — not real-world evidence.")
            st.write(item.relevant_passage)
            st.caption(f"Source: {item.source_title}")
            _source_link(item.source_url)

    st.subheader("Evidence-linked candidate explanations")
    if not result["claims"]:
        st.info("Retrieved incidents do not state a documented root cause.")
    for claim in result["claims"]:
        with st.container(border=True):
            st.write(f"**Candidate cause:** {claim.claim}")
            st.write(f"**Evidence check:** {claim.support_status} — {claim.explanation}")
            st.caption(f"Incident(s): {', '.join(claim.evidence_incident_ids) or 'none'}")
            if claim.source_passage:
                st.write(f"> {claim.source_passage}")
            st.caption(f"Source reference: {claim.source_reference or 'not provided'}")

    st.subheader("Failure-chain elements")
    if not result["failure_chains"]:
        st.info("No chain elements were available.")
    for incident_id in dict.fromkeys(node["incident_id"] for node in result["failure_chains"]):
        chain_nodes = [node for node in result["failure_chains"] if node["incident_id"] == incident_id]
        st.markdown(f"**{incident_id}**")
        for node in chain_nodes:
            label = f"{node['element']}: {node['text']}  \nEvidence: {node['evidence_status']}"
            st.write(label)

    if result["generated_analysis"]:
        st.subheader(f"Local model narrative ({model})")
        st.text(result["generated_analysis"])
        st.caption("Model text is unverified. Use the cited incident passages and evidence checks above.")
    elif result["model_warning"]:
        st.error(result["model_warning"])

    with st.expander("Missing information to collect"):
        st.markdown(
            "- Equipment make/model, age, and maintenance history\n"
            "- Operating load, environment, and symptom timeline\n"
            "- Recent changes, inspection readings, and instrument calibration\n"
            "- Applicable procedures, alarm limits, and primary incident sources"
        )


render_app()
