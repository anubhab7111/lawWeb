"""
Lawyer recommendation: pgvector semantic search over lawyer bios, fused with
a weighted score on rating/success_rate. Backs both POST /api/lawyers/recommend
and the chatbot's find_lawyer intent, so there is one recommendation
implementation instead of two divergent ones.

Reuses the embedding model already loaded for the RAG pipeline
(BAAI/bge-large-en-v1.5 via base_legal_rag._get_shared_embeddings) rather than
loading a second copy.

There is no lawyer create/update endpoint in this repo today, so embeddings
are only ever backfilled (see backfill_lawyer_embeddings.py). If a lawyer
create/update route is added later, it must call embed_lawyers_batch (or
embed_query_text on a single bio) and store the result in bio_embedding, or
new/edited lawyers will silently be excluded from semantic search.
"""

import asyncio
from typing import List, Optional, Tuple

from sqlmodel import Session, select

from app.db.models import Lawyer
from app.tools.base_legal_rag import _get_shared_embeddings

# Static specialization list surfaced by GET /api/chat/specializations.
# Relocated from the retired LawyerFinder.SPECIALIZATIONS.
LEGAL_SPECIALIZATIONS: List[str] = [
    "Criminal Defense",
    "Family Law",
    "Personal Injury",
    "Immigration",
    "Corporate Law",
    "Real Estate",
    "Intellectual Property",
    "Employment Law",
    "Tax Law",
    "Estate Planning",
    "Civil Rights",
    "Environmental Law",
    "Bankruptcy",
    "Medical Malpractice",
    "Contract Law",
]

# Weighted-score components. Semantic similarity dominates when a
# problem_description is given; when it isn't, these are renormalized to
# just rating/success_rate (see recommend_lawyers).
SEMANTIC_WEIGHT = 0.5
RATING_WEIGHT = 0.3
SUCCESS_RATE_WEIGHT = 0.2

# Broad candidate set pulled by SQL (vector order or rating order), then
# re-ranked in Python by the weighted score. At current table sizes this is
# a plain sequential scan (sub-millisecond); once the lawyers table grows
# past ~50k-100k rows, add
#   CREATE INDEX ... ON lawyers USING hnsw (bio_embedding vector_cosine_ops)
# as a pure scaling change — no query code above this needs to change.
CANDIDATE_POOL_SIZE = 20
RESULT_LIMIT = 5


def _build_embedding_text(specialty: str, bio: str) -> str:
    return f"{specialty}. {bio}"


async def embed_query_text(text: str) -> List[float]:
    """Embed free text (e.g. a user's problem description) for query-time search."""
    embeddings = await _get_shared_embeddings()
    return await embeddings.aembed_query(text)


async def embed_lawyers_batch(
    specialties_and_bios: List[Tuple[str, str]],
) -> List[List[float]]:
    """Embed (specialty, bio) pairs in one batch. Used by the backfill script."""
    embeddings = await _get_shared_embeddings()
    texts = [_build_embedding_text(specialty, bio) for specialty, bio in specialties_and_bios]
    return await asyncio.to_thread(embeddings.embed_documents, texts)


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 1.0
    return (value - lo) / (hi - lo)


async def recommend_lawyers(
    session: Session,
    *,
    problem_description: Optional[str] = None,
    specialty: Optional[str] = None,
    location: Optional[str] = None,
    max_hourly_rate: Optional[int] = None,
    min_rating: Optional[float] = None,
    limit: int = RESULT_LIMIT,
) -> List[Lawyer]:
    """
    Retrieve a candidate pool (semantic search when problem_description is
    given, otherwise rating-ordered) filtered by the structured criteria,
    then re-rank the pool by a weighted score of semantic similarity,
    rating, and success_rate.
    """
    filters = []
    if specialty:
        filters.append(Lawyer.specialty.ilike(f"%{specialty}%"))
    if location:
        filters.append(Lawyer.location.ilike(f"%{location}%"))
    if max_hourly_rate is not None:
        filters.append(Lawyer.hourly_rate <= max_hourly_rate)
    if min_rating is not None:
        filters.append(Lawyer.rating >= min_rating)

    semantic_available = bool(problem_description)

    if semantic_available:
        query_vector = await embed_query_text(problem_description)
        distance_col = Lawyer.bio_embedding.cosine_distance(query_vector).label("distance")
        stmt = select(Lawyer, distance_col).where(Lawyer.bio_embedding.is_not(None))
        for f in filters:
            stmt = stmt.where(f)
        stmt = stmt.order_by(distance_col).limit(CANDIDATE_POOL_SIZE)
        candidates: List[Tuple[Lawyer, Optional[float]]] = [
            (lawyer, float(distance)) for lawyer, distance in session.exec(stmt).all()
        ]
    else:
        stmt = select(Lawyer)
        for f in filters:
            stmt = stmt.where(f)
        stmt = stmt.order_by(Lawyer.rating.desc()).limit(CANDIDATE_POOL_SIZE)
        candidates = [(lawyer, None) for lawyer in session.exec(stmt).all()]

    if not candidates:
        return []

    ratings = [lawyer.rating for lawyer, _ in candidates]
    success_rates = [lawyer.success_rate for lawyer, _ in candidates]
    rating_lo, rating_hi = min(ratings), max(ratings)
    success_lo, success_hi = min(success_rates), max(success_rates)

    if semantic_available:
        semantic_weight, rating_weight, success_weight = (
            SEMANTIC_WEIGHT,
            RATING_WEIGHT,
            SUCCESS_RATE_WEIGHT,
        )
    else:
        # No semantic signal to blend in — renormalize onto the remaining
        # components instead of silently discarding half the score.
        remaining = RATING_WEIGHT + SUCCESS_RATE_WEIGHT
        semantic_weight = 0.0
        rating_weight = RATING_WEIGHT / remaining
        success_weight = SUCCESS_RATE_WEIGHT / remaining

    scored: List[Tuple[float, Lawyer]] = []
    for lawyer, distance in candidates:
        # Cosine distance is in [0, 2]; similarity = 1 - distance, clamped.
        similarity = max(0.0, min(1.0, 1 - distance)) if distance is not None else 0.0
        norm_rating = _normalize(lawyer.rating, rating_lo, rating_hi)
        norm_success = _normalize(lawyer.success_rate, success_lo, success_hi)
        score = (
            semantic_weight * similarity
            + rating_weight * norm_rating
            + success_weight * norm_success
        )
        scored.append((score, lawyer))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [lawyer for _, lawyer in scored[:limit]]


def format_lawyer_results(lawyers: List[Lawyer]) -> str:
    """Format lawyer results as a markdown block for the chatbot's reply."""
    if not lawyers:
        return "No lawyers found matching your criteria. Please try broadening your search."

    result_parts = [f"## Found {len(lawyers)} Lawyer(s)\n"]

    for i, lawyer in enumerate(lawyers, 1):
        result_parts.append(
            f"""
### {i}. {lawyer.name}
- **Specialty:** {lawyer.specialty}
- **Location:** {lawyer.location}
- **Experience:** {lawyer.experience} years
- **Rating:** {'⭐' * int(round(lawyer.rating))} ({lawyer.rating}/5.0)
- **Success Rate:** {lawyer.success_rate}%
- **Hourly Rate:** ${lawyer.hourly_rate}/hr
- **Availability:** {lawyer.availability}
- **Bio:** {lawyer.bio}
"""
        )

    result_parts.append(
        "\n*Note: Always verify lawyer credentials before engaging their services.*"
    )

    return "".join(result_parts)
