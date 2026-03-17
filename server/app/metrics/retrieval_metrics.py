"""
Retrieval Metrics
=================
Implements the three core search-quality metrics for the legal RAG pipeline.

Metrics
-------
1. Hit Rate @ k      — What percentage of queries have at least one relevant
                       document in the top-k retrieved results?
                       Range: [0, 1].  Target: ≥ 0.70 for k=3.

2. Mean Reciprocal   — For each query, 1 / rank of the *first* relevant result.
   Rank  (MRR)         Averaged across all queries.
                       Range: [0, 1].  MRR = 1.0 means the right document is
                       always ranked #1.

3. Context Precision — What fraction of the retrieved context is genuinely
                       relevant to the query?  Evaluated by the LLM judge
                       (see llm_judge.py) when a judge instance is provided,
                       or approximated via keyword overlap when running offline.
                       Range: [0, 1].

All pure-metric functions accept plain Python lists and return plain floats so
they can be unit-tested without any external dependencies.  The LLM-judge path
is wired in only through the optional ``judge`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RetrievalSample:
    """
    All the information needed to evaluate retrieval quality for one query.

    Attributes
    ----------
    query:
        The exact user query string.
    retrieved_sections:
        Ordered list of IPC section numbers (or document IDs) that the RAG
        system returned, ranked from most to least similar.
        e.g. ``["420", "406", "302", "465", "468"]``
    retrieved_context:
        Raw text of all retrieved chunks concatenated.  Used by Context
        Precision (LLM judge) and as a fallback for keyword-based scoring.
    relevant_sections:
        Ground-truth set of section numbers / doc-IDs that are considered
        correct for this query.  Comes from :data:`ground_truth.GROUND_TRUTH`.
    relevant_keywords:
        Fallback list of keywords; used when ``relevant_sections`` is empty
        (e.g. constitutional queries where the IPC RAG is not expected to
        retrieve anything) to compute a keyword-based precision proxy.
    """

    query: str
    retrieved_sections: List[str]
    retrieved_context: str
    relevant_sections: List[str]
    relevant_keywords: List[str] = field(default_factory=list)


@dataclass
class RetrievalMetricsResult:
    """Aggregated retrieval metrics for a full evaluation run."""

    # Per-query raw data
    samples: List[RetrievalSample]

    # Hit Rate @ various k
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float

    # MRR
    mrr: float

    # Context Precision (average across all queries)
    context_precision_mean: float
    context_precision_per_query: List[float]

    # Diagnostics
    queries_with_relevant_sections: int  # how many GT entries have IPC sections
    queries_with_context: int  # how many retrieved non-empty context

    def to_dict(self) -> Dict[str, float]:
        """Flat dictionary suitable for CSV / JSON export."""
        return {
            "hit_rate_at_1": self.hit_rate_at_1,
            "hit_rate_at_3": self.hit_rate_at_3,
            "hit_rate_at_5": self.hit_rate_at_5,
            "mrr": self.mrr,
            "context_precision_mean": self.context_precision_mean,
            "queries_with_relevant_sections": float(
                self.queries_with_relevant_sections
            ),
            "queries_with_context": float(self.queries_with_context),
        }


# ---------------------------------------------------------------------------
# Core scalar metrics  (pure functions — no I/O)
# ---------------------------------------------------------------------------


def compute_hit_rate(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int = 5,
) -> float:
    """
    Compute Hit Rate @ k for a *single* query.

    A "hit" occurs when at least one element from *relevant* appears in the
    first *k* items of *retrieved*.

    Args:
        retrieved: Ordered sequence of retrieved document IDs / section numbers.
                   The first element is the highest-ranked result.
        relevant:  Set of ground-truth relevant IDs for this query.
        k:         Cut-off rank.  Defaults to 5.

    Returns:
        1.0 if any relevant document appears in top-k, else 0.0.

    Examples:
        >>> compute_hit_rate(["420", "406", "302"], ["420"], k=3)
        1.0
        >>> compute_hit_rate(["302", "380", "406"], ["420"], k=3)
        0.0
        >>> compute_hit_rate([], ["420"], k=3)
        0.0
        >>> compute_hit_rate(["420"], [], k=3)  # no ground truth → skip
        0.0
    """
    if not relevant or not retrieved:
        return 0.0
    relevant_set = set(str(r).strip() for r in relevant)
    top_k = [str(r).strip() for r in retrieved[:k]]
    return 1.0 if relevant_set.intersection(top_k) else 0.0


def compute_mrr_single(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """
    Compute the Reciprocal Rank for a *single* query.

    Reciprocal Rank = 1 / (rank of first relevant document).
    If no relevant document is found in the retrieved list, the score is 0.0.

    Args:
        retrieved: Ordered sequence of retrieved document IDs (highest rank first).
        relevant:  Ground-truth relevant IDs for this query.

    Returns:
        Float in (0, 1].  Returns 0.0 when no relevant document is found.

    Examples:
        >>> compute_mrr_single(["420", "406", "302"], ["420"])
        1.0
        >>> compute_mrr_single(["302", "420", "406"], ["420"])
        0.5
        >>> compute_mrr_single(["302", "380", "465"], ["420"])
        0.0
    """
    if not relevant or not retrieved:
        return 0.0
    relevant_set = set(str(r).strip() for r in relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if str(doc_id).strip() in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_mrr(samples: Sequence[RetrievalSample]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over a collection of retrieval samples.

    Only samples that have non-empty ``relevant_sections`` contribute to the
    mean (queries where the IPC RAG is not expected to surface anything are
    excluded, rather than penalising the MRR unfairly).

    Args:
        samples: Collection of :class:`RetrievalSample` objects.

    Returns:
        Mean Reciprocal Rank as a float in [0, 1].  Returns 0.0 for an empty
        or all-excluded sample set.

    Example:
        >>> s1 = RetrievalSample("q1", ["420", "406"], "", ["420"])
        >>> s2 = RetrievalSample("q2", ["302", "420"], "", ["420"])
        >>> compute_mrr([s1, s2])
        0.75
    """
    rrs: List[float] = []
    for sample in samples:
        if not sample.relevant_sections:
            # This query has no IPC ground truth — skip for MRR
            continue
        rr = compute_mrr_single(sample.retrieved_sections, sample.relevant_sections)
        rrs.append(rr)

    if not rrs:
        return 0.0
    return round(sum(rrs) / len(rrs), 4)


def compute_hit_rate_at_k(
    samples: Sequence[RetrievalSample],
    k: int = 5,
) -> float:
    """
    Compute Hit Rate @ k averaged over all samples that have ground truth.

    Args:
        samples: Collection of :class:`RetrievalSample` objects.
        k:       Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    hits: List[float] = []
    for sample in samples:
        if not sample.relevant_sections:
            continue  # No IPC ground truth for this query
        hit = compute_hit_rate(sample.retrieved_sections, sample.relevant_sections, k=k)
        hits.append(hit)

    if not hits:
        return 0.0
    return round(sum(hits) / len(hits), 4)


# ---------------------------------------------------------------------------
# Context Precision  (keyword-based fallback — no LLM required)
# ---------------------------------------------------------------------------


def _keyword_precision_score(context: str, keywords: List[str]) -> float:
    """
    Lightweight keyword-overlap proxy for Context Precision.

    Used when:
      • No LLM judge is available (offline / unit-test mode).
      • The query has no relevant IPC sections (constitutional queries) and
        we still want a numeric precision signal.

    Algorithm:
      For each keyword in *keywords*, check whether it appears (case-insensitive)
      in *context*.  Score = matched_keywords / total_keywords.

    Args:
        context:  Concatenated retrieved text.
        keywords: Expected keywords from the ground-truth entry.

    Returns:
        Float in [0, 1].  Returns 0.0 if context or keywords are empty.
    """
    if not context or not keywords:
        return 0.0
    context_lower = context.lower()
    matched = sum(1 for kw in keywords if kw.lower() in context_lower)
    return round(matched / len(keywords), 4)


def compute_context_precision(
    query: str,
    context: str,
    keywords: Optional[List[str]] = None,
    judge=None,  # Optional[LLMJudge]  — avoid hard import for offline use
) -> float:
    """
    Compute Context Precision for a single query (synchronous entry-point).

    Strategy
    --------
    * If *judge* is provided → delegates to :meth:`LLMJudge.context_precision`
      (async; called via ``asyncio.run`` if no event loop is active, or
      awaited if one is running).
    * Otherwise → falls back to keyword-overlap scoring using *keywords*.

    Args:
        query:    The user query string.
        context:  All retrieved text chunks concatenated.
        keywords: Ground-truth keywords to match against (fallback mode).
        judge:    Optional :class:`~app.metrics.llm_judge.LLMJudge` instance.

    Returns:
        Float in [0, 1].
    """
    if not context:
        return 0.0

    if judge is not None:
        # Run the async judge in whichever event-loop context we are in
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are already inside an async context — create a task
                # Callers should use compute_context_precision_async instead
                logger.debug(
                    "compute_context_precision called synchronously from async "
                    "context; falling back to keyword scoring."
                )
            else:
                score_obj = loop.run_until_complete(
                    judge.context_precision(query=query, context=context)
                )
                return round(score_obj.score, 4)
        except Exception as exc:
            logger.warning(
                "LLM judge failed for context_precision: %s — falling back.", exc
            )

    # Keyword fallback
    return _keyword_precision_score(context, keywords or [])


async def compute_context_precision_async(
    query: str,
    context: str,
    keywords: Optional[List[str]] = None,
    judge=None,  # Optional[LLMJudge]
) -> float:
    """
    Async version of :func:`compute_context_precision`.

    Prefer this function when you are already inside an ``async`` function
    (e.g. the evaluator loop).

    Args:
        query:    The user query string.
        context:  All retrieved text chunks concatenated.
        keywords: Ground-truth keywords (fallback when judge is None).
        judge:    Optional :class:`~app.metrics.llm_judge.LLMJudge` instance.

    Returns:
        Float in [0, 1].
    """
    if not context:
        return 0.0

    if judge is not None:
        try:
            score_obj = await judge.context_precision(query=query, context=context)
            return round(score_obj.score, 4)
        except Exception as exc:
            logger.warning(
                "LLM judge async failed for context_precision: %s — falling back.",
                exc,
            )

    return _keyword_precision_score(context, keywords or [])


# ---------------------------------------------------------------------------
# Aggregate: compute all retrieval metrics in one call
# ---------------------------------------------------------------------------


async def compute_all_retrieval_metrics(
    samples: Sequence[RetrievalSample],
    judge=None,  # Optional[LLMJudge]
) -> RetrievalMetricsResult:
    """
    Compute Hit Rate @ 1/3/5, MRR, and Context Precision for a full batch.

    Context Precision for each sample is computed in parallel (all samples
    whose context is non-empty), using the LLM judge when available and
    keyword scoring otherwise.

    Args:
        samples: Sequence of :class:`RetrievalSample` objects — one per query.
        judge:   Optional :class:`~app.metrics.llm_judge.LLMJudge`.

    Returns:
        :class:`RetrievalMetricsResult` with all metrics populated.
    """
    samples = list(samples)

    # ── Hit Rate ─────────────────────────────────────────────────────────────
    hr1 = compute_hit_rate_at_k(samples, k=1)
    hr3 = compute_hit_rate_at_k(samples, k=3)
    hr5 = compute_hit_rate_at_k(samples, k=5)

    # ── MRR ──────────────────────────────────────────────────────────────────
    mrr = compute_mrr(samples)

    # ── Context Precision (parallel) ─────────────────────────────────────────
    precision_tasks = [
        compute_context_precision_async(
            query=s.query,
            context=s.retrieved_context,
            keywords=s.relevant_keywords,
            judge=judge,
        )
        for s in samples
    ]
    context_precision_per_query: List[float] = list(
        await asyncio.gather(*precision_tasks, return_exceptions=False)
    )

    cp_mean = (
        round(sum(context_precision_per_query) / len(context_precision_per_query), 4)
        if context_precision_per_query
        else 0.0
    )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    queries_with_gt = sum(1 for s in samples if s.relevant_sections)
    queries_with_ctx = sum(1 for s in samples if s.retrieved_context.strip())

    return RetrievalMetricsResult(
        samples=list(samples),
        hit_rate_at_1=hr1,
        hit_rate_at_3=hr3,
        hit_rate_at_5=hr5,
        mrr=mrr,
        context_precision_mean=cp_mean,
        context_precision_per_query=context_precision_per_query,
        queries_with_relevant_sections=queries_with_gt,
        queries_with_context=queries_with_ctx,
    )


# ---------------------------------------------------------------------------
# Pretty-printer helper
# ---------------------------------------------------------------------------


def print_retrieval_report(result: RetrievalMetricsResult) -> None:
    """Print a formatted retrieval metrics summary to stdout."""
    sep = "─" * 52
    print(f"\n{sep}")
    print("  RETRIEVAL METRICS")
    print(sep)
    print(f"  Queries evaluated            : {len(result.samples)}")
    print(f"  Queries with IPC ground truth: {result.queries_with_relevant_sections}")
    print(f"  Queries with retrieved ctx   : {result.queries_with_context}")
    print(sep)
    print(f"  Hit Rate @ 1                 : {result.hit_rate_at_1:.4f}")
    print(f"  Hit Rate @ 3                 : {result.hit_rate_at_3:.4f}")
    print(f"  Hit Rate @ 5                 : {result.hit_rate_at_5:.4f}")
    print(sep)
    print(f"  Mean Reciprocal Rank (MRR)   : {result.mrr:.4f}")
    print(sep)
    print(f"  Context Precision (mean)     : {result.context_precision_mean:.4f}")
    print(sep)

    # Per-query breakdown
    if result.samples:
        print("\n  Per-query precision breakdown:")
        print(f"  {'#':<4} {'Precision':<12} {'Query (truncated)'}")
        print(f"  {'':-<4} {'':-<12} {'':-<45}")
        for i, (sample, prec) in enumerate(
            zip(result.samples, result.context_precision_per_query), start=1
        ):
            q_short = (
                (sample.query[:45] + "…") if len(sample.query) > 46 else sample.query
            )
            print(f"  {i:<4} {prec:<12.4f} {q_short}")
    print()
