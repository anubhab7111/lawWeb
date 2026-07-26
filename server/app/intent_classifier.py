"""
Reference-example embedding classifier for chatbot intent routing.

Replaces an LLM-based structured router with a nearest-centroid classifier
over the same BGE embedding model already loaded for RAG retrieval
(app.tools.base_legal_rag._get_shared_embeddings) — zero additional
VRAM/RAM cost versus today, no Ollama round-trip, no multi-second latency.

Scope: classifies among the four intents that survive app.chatbot's stage-1
non-legal short-circuit (_stage1_domain_check runs first, zero latency, and
filters "non_legal"). This module never represents "non_legal" as a class —
by construction it's only ever called on input stage-1 already judged legal.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.tools.base_legal_rag import _get_shared_embeddings

# ============================================================================
# Reference example set — the classifier's seed data. Kept deliberately
# separate from any eval/ground-truth set (see server/test_routing.py):
# testing a nearest-neighbor classifier against the literal strings it was
# built from would be close to meaningless.
# ============================================================================

INTENT_REFERENCE_EXAMPLES: Dict[str, List[str]] = {
    "document_analysis": [
        "Can you review this rental agreement for any issues?",
        "Please check if this contract is legally valid.",
        "What are the key points in this document I uploaded?",
        "Does this will comply with the legal requirements?",
        "I've attached my employment contract, can you analyze it?",
        "Check this document for missing clauses or defects.",
        "Summarize the legal implications of this agreement.",
        "Is this notice properly drafted according to law?",
    ],
    "crime_report": [
        "Someone stole my phone yesterday, what should I do?",
        "I was scammed out of money by an online seller.",
        "My neighbor threatened me with violence, how do I report this?",
        "I was assaulted outside my house last night.",
        "Someone broke into my house and took my belongings.",
        "I've been a victim of credit card fraud, please help.",
        "My employer threatened me when I asked for unpaid wages.",
        "I want to file an FIR against someone who cheated me.",
    ],
    "find_lawyer": [
        "I need a lawyer for a divorce case in Mumbai.",
        "Can you recommend a criminal defense attorney near me?",
        "Find me a property lawyer in Delhi.",
        "I'm looking for legal representation for a business dispute.",
        "Who is a good family lawyer in Bangalore?",
        "I need to hire an advocate for a court case.",
        "Suggest some lawyers who specialize in labor law.",
        "Where can I find a lawyer to help with my case?",
    ],
    "general_query": [
        "What are the grounds for divorce under Hindu law?",
        "Which IPC sections apply to cyberstalking?",
        "Is an oral contract legally binding in India?",
        "What is the punishment for theft under IPC?",
        "Explain the right to privacy under the Constitution.",
        "What are my rights as a tenant if my landlord wants to evict me?",
        "Can a company terminate an employee without notice?",
        "What is the procedure for filing a consumer complaint?",
        "What does Article 21 of the Constitution protect?",
        "How does bail work for a non-bailable offence?",
    ],
}

# ============================================================================
# Tuning constants
# ============================================================================

TOP_K_MEAN = 3  # aggregate = mean of the top-3 example similarities
DOCUMENT_ANALYSIS_BOOST = 0.15  # additive prior on document_analysis when has_document
AMBIGUITY_MARGIN = 0.03  # top-vs-runner-up margin below which routing is ambiguous
MIN_CONFIDENT_SCORE = 0.35  # top score below this is ambiguous regardless of margin
SECONDARY_INTENT_MARGIN = 0.05  # runner-ups within this of top score -> secondary_intents


@dataclass
class IntentClassification:
    """Plain dataclass, not Pydantic — there's no LLM structured-output
    target to validate against, so schema/validation machinery has nothing
    left to do."""

    primary_intent: str
    confidence: float  # top intent's aggregated similarity (not a probability)
    margin: float  # top score minus runner-up score
    is_ambiguous: bool
    reasoning: str
    secondary_intents: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)  # all intents, for logs/debug


# ============================================================================
# Lazy singleton: embed the reference set exactly once per process
# (mirrors _get_shared_embeddings' double-checked-locking pattern)
# ============================================================================

_reference_embeddings: Optional[Dict[str, List[List[float]]]] = None
_reference_embeddings_lock = asyncio.Lock()


async def _get_reference_embeddings() -> Dict[str, List[List[float]]]:
    global _reference_embeddings
    if _reference_embeddings is not None:
        return _reference_embeddings
    async with _reference_embeddings_lock:
        if _reference_embeddings is None:
            embeddings = await _get_shared_embeddings()
            all_texts: List[str] = []
            spans: Dict[str, tuple] = {}
            for intent, examples in INTENT_REFERENCE_EXAMPLES.items():
                start = len(all_texts)
                all_texts.extend(examples)
                spans[intent] = (start, len(all_texts))
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None, lambda: embeddings.embed_documents(all_texts)
            )
            _reference_embeddings = {
                intent: vectors[start:end] for intent, (start, end) in spans.items()
            }
            print(
                f"[IntentClassifier] Embedded {len(all_texts)} reference "
                f"examples across {len(spans)} intents."
            )
    return _reference_embeddings


def _dot(a: List[float], b: List[float]) -> float:
    # Embeddings are pre-normalized (normalize_embeddings=True in
    # base_legal_rag.py), so dot product == cosine similarity.
    return sum(x * y for x, y in zip(a, b))


def _aggregate(query_vec: List[float], example_vecs: List[List[float]]) -> float:
    """Top-K mean: less fragile than a single max (one oddly-phrased example
    spiking the score), more discriminating than a full mean over 8-10
    examples spanning a broad intent like general_query."""
    sims = sorted((_dot(query_vec, v) for v in example_vecs), reverse=True)
    top = sims[:TOP_K_MEAN]
    return sum(top) / len(top)


async def classify_intent_embedding(
    text: str, has_document: bool
) -> IntentClassification:
    embeddings = await _get_shared_embeddings()
    reference = await _get_reference_embeddings()
    loop = asyncio.get_event_loop()
    query_vec = await loop.run_in_executor(None, lambda: embeddings.embed_query(text))

    scores: Dict[str, float] = {
        intent: _aggregate(query_vec, vecs) for intent, vecs in reference.items()
    }
    if has_document:
        scores["document_analysis"] += DOCUMENT_ANALYSIS_BOOST

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else top_score
    margin = top_score - runner_up_score

    secondary = [
        intent
        for intent, score in ranked[1:]
        if (top_score - score) <= SECONDARY_INTENT_MARGIN
    ]
    is_ambiguous = margin < AMBIGUITY_MARGIN or top_score < MIN_CONFIDENT_SCORE

    return IntentClassification(
        primary_intent=top_intent,
        confidence=top_score,
        margin=margin,
        is_ambiguous=is_ambiguous,
        reasoning=(
            f"embedding nearest-centroid: top={top_intent}({top_score:.3f}), "
            f"margin={margin:.3f}"
        ),
        secondary_intents=secondary,
        scores=scores,
    )
