"""Retrieval Metrics ================= Implements the three core search-quality metrics for the legal RAG pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class RetrievalSample:
    """All the information needed to evaluate retrieval quality for one query."""

    query: str
    retrieved_sections: List[str]
    retrieved_context: str
    relevant_sections: List[str]
    relevant_keywords: List[str] = field(default_factory=list)


@dataclass
class RetrievalMetricsResult:
    """Aggregated retrieval metrics for a full evaluation run."""

    samples: List[RetrievalSample]

    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float

    mrr: float

    context_precision_mean: float
    context_precision_per_query: List[float]

    queries_with_relevant_sections: int
    queries_with_context: int

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


def act_aware_sections(
    retrieved: Sequence[Sequence[str]], expected_acts: Sequence[str]
) -> List[str]:
    """Section numbers from (act, section) pairs, keeping rank positions but blanking any hit from an Act the ground truth doesn't expect — so "420" in the wrong statute no longer..."""
    from app.tools.case_law_rag import _acts_match, normalize_act

    wanted = [normalize_act(a) for a in expected_acts if a]
    sections = [str(sec) for _, sec in retrieved]
    if not wanted:
        return sections
    keep = [
        any(_acts_match(w, normalize_act(act)) for w in wanted) for act, _ in retrieved
    ]
    if not any(keep):
        return sections
    return [sec if ok else "" for sec, ok in zip(sections, keep)]


def compute_hit_rate(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int = 5,
) -> float:
    """Compute Hit Rate @ k for a *single* query."""
    if not relevant or not retrieved:
        return 0.0
    relevant_set = set(str(r).strip() for r in relevant)
    top_k = [str(r).strip() for r in retrieved[:k]]
    return 1.0 if relevant_set.intersection(top_k) else 0.0


def compute_mrr_single(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """Compute the Reciprocal Rank for a *single* query."""
    if not relevant or not retrieved:
        return 0.0
    relevant_set = set(str(r).strip() for r in relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if str(doc_id).strip() in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_mrr(samples: Sequence[RetrievalSample]) -> float:
    """Compute Mean Reciprocal Rank (MRR) over a collection of retrieval samples."""
    rrs: List[float] = []
    for sample in samples:
        if not sample.relevant_sections:

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
    """Compute Hit Rate @ k averaged over all samples that have ground truth."""
    hits: List[float] = []
    for sample in samples:
        if not sample.relevant_sections:
            continue
        hit = compute_hit_rate(sample.retrieved_sections, sample.relevant_sections, k=k)
        hits.append(hit)

    if not hits:
        return 0.0
    return round(sum(hits) / len(hits), 4)


def _keyword_precision_score(context: str, keywords: List[str]) -> float:
    """Lightweight keyword-overlap proxy for Context Precision."""
    if not context or not keywords:
        return 0.0
    context_lower = context.lower()
    matched = sum(1 for kw in keywords if kw.lower() in context_lower)
    return round(matched / len(keywords), 4)


def compute_context_precision(
    query: str,
    context: str,
    keywords: Optional[List[str]] = None,
    judge=None,
) -> float:
    """Compute Context Precision for a single query (synchronous entry-point)."""
    if not context:
        return 0.0

    if judge is not None:

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():

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

    return _keyword_precision_score(context, keywords or [])


async def compute_context_precision_async(
    query: str,
    context: str,
    keywords: Optional[List[str]] = None,
    judge=None,
) -> float:
    """Async version of :func:`compute_context_precision`."""
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


async def compute_all_retrieval_metrics(
    samples: Sequence[RetrievalSample],
    judge=None,
) -> RetrievalMetricsResult:
    """Compute Hit Rate @ 1/3/5, MRR, and Context Precision for a full batch."""
    samples = list(samples)

    hr1 = compute_hit_rate_at_k(samples, k=1)
    hr3 = compute_hit_rate_at_k(samples, k=3)
    hr5 = compute_hit_rate_at_k(samples, k=5)

    mrr = compute_mrr(samples)

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
