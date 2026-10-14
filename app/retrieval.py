from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import IncidentRecord, SearchResultItem

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def incident_search_text(incident: IncidentRecord) -> str:
    return " ".join(
        filter(
            None,
            [
                incident.title,
                incident.description,
                incident.system_type,
                incident.domain,
                " ".join(incident.observed_symptoms),
                " ".join(incident.contributing_conditions),
                incident.documented_root_cause or "",
                incident.failure_mechanism or "",
                incident.outcome_or_impact or "",
                " ".join(incident.corrective_actions),
                " ".join(incident.preventive_actions),
                incident.relevant_passage or "",
            ],
        )
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    tokenized = [_tokens(doc) for doc in documents]
    query_tokens = _tokens(query)
    if not tokenized or not query_tokens:
        return [0.0] * len(documents)

    doc_lengths = [len(doc) for doc in tokenized]
    average_length = sum(doc_lengths) / len(doc_lengths) or 1.0
    frequencies = [Counter(doc) for doc in tokenized]
    document_frequency = Counter(token for doc in frequencies for token in doc)
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_lengths[index] / average_length)
            score += idf * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _rank(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def search_incidents(
    query: str,
    incidents: Iterable[IncidentRecord],
    top_k: int = 5,
    domain: str | None = None,
    system_type: str | None = None,
    method: str = "hybrid",
) -> list[SearchResultItem]:
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if len(query) > 2000:
        raise ValueError("Search query exceeds 2000 characters.")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")
    if method not in {"hybrid", "vector"}:
        raise ValueError("method must be 'hybrid' or 'vector'.")

    candidates = [
        incident
        for incident in incidents
        if (domain is None or incident.domain.casefold() == domain.casefold())
        and (system_type is None or incident.system_type.casefold() == system_type.casefold())
    ]
    if not candidates:
        return []

    documents = [incident_search_text(item) for item in candidates]
    # This local TF-IDF vector index is an explicit no-download prototype baseline,
    # not a pretrained semantic embedding model.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents + [query])
    vector_scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel().tolist()
    vector_order = _rank(vector_scores)

    if method == "vector":
        order = vector_order
        score_by_index = {index: vector_scores[index] for index in order}
    else:
        bm25_scores = _bm25_scores(query, documents)
        bm25_order = _rank(bm25_scores)
        rrf_scores: dict[int, float] = {}
        for ranking in (vector_order, bm25_order):
            for rank, index in enumerate(ranking, start=1):
                rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (60 + rank)
        order = sorted(rrf_scores, key=lambda index: (-rrf_scores[index], index))
        score_by_index = rrf_scores

    results: list[SearchResultItem] = []
    query_domain = domain.casefold() if domain else None
    for index in order[:top_k]:
        item = candidates[index]
        passage = item.relevant_passage or item.description
        results.append(
            SearchResultItem(
                incident_id=item.incident_id,
                domain=item.domain,
                system_type=item.system_type,
                title=item.title,
                source_title=item.source_title,
                source_url=item.source_url,
                score=round(float(score_by_index[index]), 6),
                provenance=item.provenance,
                relevant_passage=passage,
                same_domain=query_domain == item.domain.casefold() if query_domain else False,
            )
        )
    return results
