"""
Generation Metrics  (The RAG Triad)
====================================
Implements the three core generation-quality metrics for a RAG pipeline:

  1. Faithfulness      — What fraction of the answer's claims are grounded in the
                         retrieved context?  (anti-hallucination metric)
  2. Answer Relevance  — Does the answer actually address the user's question?
  3. Context Recall    — Does the answer cover the key facts present in the
                         retrieved documents / reference answer?

Each function returns a :class:`GenerationScore` that wraps the raw
:class:`~app.metrics.llm_judge.JudgeScore` with additional bookkeeping
(query id, domain, lightweight keyword fallback, etc.).

Design notes
------------
* All metrics delegate scoring to :class:`~app.metrics.llm_judge.LLMJudge`
  (LLM-as-a-judge using the local Ollama model).
* A lightweight **keyword fallback** is always computed and stored alongside
  the LLM score.  If the LLM judge is unavailable the fallback score is used
  instead, keeping the evaluation suite runnable even without a live model.
* Functions accept plain strings so they can be used independently of the
  rest of the chatbot infrastructure (unit-testable).
* All public functions are async; they can be gathered with asyncio to run
  multiple metrics in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.metrics.llm_judge import JudgeScore, LLMJudge, get_judge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GenerationScore:
    """
    Enriched result for a single generation metric evaluation.

    Attributes:
        metric:         Name of the metric (faithfulness / answer_relevance /
                        context_recall).
        llm_score:      Score from the LLM judge in [0, 1].
        keyword_score:  Lightweight keyword-based fallback score in [0, 1].
        final_score:    The score that should be reported.  Equal to
                        *llm_score* when the judge succeeded, else falls back
                        to *keyword_score*.
        reasoning:      Human-readable explanation from the judge.
        judge_failed:   True when the LLM judge could not produce a score and
                        the keyword fallback was used instead.
        judge_latency_s: Wall-clock time the LLM judge took (0 if skipped).
    """

    metric: str
    llm_score: float
    keyword_score: float
    final_score: float
    reasoning: str
    judge_failed: bool = False
    judge_latency_s: float = 0.0

    @property
    def label(self) -> str:
        """Human-readable quality band based on final_score."""
        s = self.final_score
        if s >= 0.85:
            return "Excellent"
        if s >= 0.70:
            return "Good"
        if s >= 0.50:
            return "Fair"
        if s >= 0.30:
            return "Poor"
        return "Very Poor"

    def __repr__(self) -> str:  # noqa: D105
        src = "llm" if not self.judge_failed else "keyword-fallback"
        return (
            f"GenerationScore(metric={self.metric!r}, "
            f"final={self.final_score:.2f} [{self.label}], src={src})"
        )


# ---------------------------------------------------------------------------
# Keyword / heuristic fallbacks
# (fast, zero-cost, used when the LLM judge is unavailable)
# ---------------------------------------------------------------------------


def _keyword_faithfulness(answer: str, context: str) -> float:
    """
    Heuristic faithfulness score.

    Looks for phrases that strongly signal hallucination (definite claims
    with no grounding) vs. grounded hedging language.  Very rough — use
    only as a fallback.
    """
    if not answer.strip() or not context.strip():
        return 0.0

    answer_lower = answer.lower()
    context_lower = context.lower()

    # Phrases that indicate cautious, grounded language → good signal
    grounding_phrases = [
        "according to",
        "as per",
        "retrieved context",
        "the retrieved",
        "ipc section",
        "section ",
        "article ",
        "as stated",
        "as mentioned",
        "the court held",
        "the act provides",
        "under",
        "pursuant to",
    ]
    grounding_hits = sum(1 for p in grounding_phrases if p in answer_lower)

    # Phrases that suggest fabrication
    hallucination_signals = [
        "section 243b",  # Fabricated section (seen in eval CSV)
        "negligence act, 1872",  # Fictitious act
        "indian negligence act",
        "section 66 of the constitution",  # Misattributed section
    ]
    hallucination_hits = sum(1 for p in hallucination_signals if p in answer_lower)

    # Compute overlap: how many key terms from the answer appear in context?
    answer_words = set(re.findall(r"\b[a-z]{5,}\b", answer_lower))
    context_words = set(re.findall(r"\b[a-z]{5,}\b", context_lower))
    overlap = len(answer_words & context_words)
    overlap_ratio = min(overlap / max(len(answer_words), 1), 1.0)

    # Base score from overlap
    base = overlap_ratio * 0.7
    # Bonus for grounding language
    base = min(base + min(grounding_hits, 5) * 0.02, 1.0)
    # Penalty for hallucination signals
    penalty = hallucination_hits * 0.15
    return round(max(0.0, min(base - penalty, 1.0)), 3)


def _keyword_answer_relevance(query: str, answer: str) -> float:
    """
    Heuristic answer relevance score.

    Measures keyword overlap between the query and the answer.  Penalises
    very short answers (< 50 chars) and error messages.
    """
    if not answer.strip():
        return 0.0
    if answer.startswith("ERROR") or "I'm having trouble" in answer:
        return 0.1

    query_lower = query.lower()
    answer_lower = answer.lower()

    # Core query terms (length ≥ 4 to skip stop words)
    query_terms = set(re.findall(r"\b[a-z]{4,}\b", query_lower))
    if not query_terms:
        return 0.5

    answer_terms = set(re.findall(r"\b[a-z]{4,}\b", answer_lower))
    matched = query_terms & answer_terms
    term_coverage = len(matched) / len(query_terms)

    # Length signal: too short → probably unhelpful
    length_score = min(len(answer) / 500, 1.0)

    # Direct answer signals
    directness_phrases = [
        "yes",
        "no",
        "under section",
        "the court",
        "is not",
        "can be",
        "is valid",
        "is void",
        "is illegal",
        "is permissible",
        "is binding",
        "is enforceable",
    ]
    directness = any(p in answer_lower[:300] for p in directness_phrases)
    directness_bonus = 0.10 if directness else 0.0

    raw = term_coverage * 0.65 + length_score * 0.25 + directness_bonus
    return round(min(raw, 1.0), 3)


def _keyword_context_recall(reference_answer: str, model_answer: str) -> float:
    """
    Heuristic context recall: keyword overlap between reference and model answer.
    """
    if not reference_answer.strip():
        return 1.0
    if not model_answer.strip():
        return 0.0

    ref_terms = set(re.findall(r"\b[a-z]{5,}\b", reference_answer.lower()))
    model_terms = set(re.findall(r"\b[a-z]{5,}\b", model_answer.lower()))

    if not ref_terms:
        return 1.0

    covered = ref_terms & model_terms
    return round(len(covered) / len(ref_terms), 3)


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


async def compute_faithfulness(
    answer: str,
    retrieved_context: str,
    judge: Optional[LLMJudge] = None,
    *,
    use_keyword_fallback: bool = True,
) -> GenerationScore:
    """
    Faithfulness (Groundedness).

    Measures what fraction of the claims in *answer* are explicitly
    supported by *retrieved_context*.  A score of 1.0 means every claim
    can be traced back to the retrieved documents — zero hallucination.

    This is the primary anti-hallucination metric for the legal RAG system.
    The LLM judge checks each factual claim in the answer against the context
    and returns a ratio of supported vs. total claims.

    Args:
        answer:             The chatbot's generated response.
        retrieved_context:  All RAG / Indian Kanoon context that was passed
                            to the LLM when generating the answer.
        judge:              Optional :class:`~app.metrics.llm_judge.LLMJudge`
                            instance.  Uses the module singleton if ``None``.
        use_keyword_fallback:
                            When ``True`` (default), falls back to the
                            keyword heuristic if the LLM judge fails.

    Returns:
        :class:`GenerationScore` with metric ``"faithfulness"``.
    """
    j = judge or get_judge()
    kw_score = (
        _keyword_faithfulness(answer, retrieved_context)
        if use_keyword_fallback
        else 0.5
    )

    try:
        result: JudgeScore = await j.faithfulness(
            answer=answer,
            context=retrieved_context,
        )
        judge_failed = result.reasoning.startswith("[JUDGE FAILED]")
        final = kw_score if judge_failed and use_keyword_fallback else result.score

        return GenerationScore(
            metric="faithfulness",
            llm_score=result.score,
            keyword_score=kw_score,
            final_score=round(final, 3),
            reasoning=result.reasoning,
            judge_failed=judge_failed,
            judge_latency_s=result.latency_s,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Faithfulness judge error: %s", exc)
        return GenerationScore(
            metric="faithfulness",
            llm_score=kw_score,
            keyword_score=kw_score,
            final_score=kw_score,
            reasoning=f"[JUDGE EXCEPTION] {exc} — keyword fallback used.",
            judge_failed=True,
        )


async def compute_answer_relevance(
    query: str,
    answer: str,
    judge: Optional[LLMJudge] = None,
    *,
    use_keyword_fallback: bool = True,
) -> GenerationScore:
    """
    Answer Relevance.

    Measures how directly and completely *answer* addresses *query*.
    A score of 1.0 means the answer fully resolves the user's legal question.
    A low score (< 0.5) means the answer is off-topic, generic, or only
    partially addresses the question — indicating the agentic routing or
    intent classification may be mis-firing.

    Args:
        query:   The user's legal question (from TEST_PROMPTS).
        answer:  The chatbot's generated response.
        judge:   Optional :class:`~app.metrics.llm_judge.LLMJudge` instance.
        use_keyword_fallback: Fall back to keyword heuristic on judge failure.

    Returns:
        :class:`GenerationScore` with metric ``"answer_relevance"``.
    """
    j = judge or get_judge()
    kw_score = _keyword_answer_relevance(query, answer) if use_keyword_fallback else 0.5

    try:
        result: JudgeScore = await j.answer_relevance(
            question=query,
            answer=answer,
        )
        judge_failed = result.reasoning.startswith("[JUDGE FAILED]")
        final = kw_score if judge_failed and use_keyword_fallback else result.score

        return GenerationScore(
            metric="answer_relevance",
            llm_score=result.score,
            keyword_score=kw_score,
            final_score=round(final, 3),
            reasoning=result.reasoning,
            judge_failed=judge_failed,
            judge_latency_s=result.latency_s,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer relevance judge error: %s", exc)
        return GenerationScore(
            metric="answer_relevance",
            llm_score=kw_score,
            keyword_score=kw_score,
            final_score=kw_score,
            reasoning=f"[JUDGE EXCEPTION] {exc} — keyword fallback used.",
            judge_failed=True,
        )


async def compute_context_recall(
    reference_answer: str,
    model_answer: str,
    judge: Optional[LLMJudge] = None,
    *,
    use_keyword_fallback: bool = True,
) -> GenerationScore:
    """
    Context Recall.

    Measures what fraction of the key legal facts in *reference_answer*
    are also present in *model_answer*.  A score of 1.0 means the model
    captured all important information from the ground-truth reference.
    A low score means critical facts (e.g. the correct section number, a
    key case name, or the correct legal test) were omitted.

    Args:
        reference_answer:  Gold-standard answer from
                           :data:`~app.metrics.ground_truth.GROUND_TRUTH`.
        model_answer:      The chatbot's generated response.
        judge:             Optional :class:`~app.metrics.llm_judge.LLMJudge`.
        use_keyword_fallback: Fall back to keyword heuristic on judge failure.

    Returns:
        :class:`GenerationScore` with metric ``"context_recall"``.
    """
    j = judge or get_judge()
    kw_score = (
        _keyword_context_recall(reference_answer, model_answer)
        if use_keyword_fallback
        else 0.5
    )

    try:
        result: JudgeScore = await j.context_recall(
            reference_answer=reference_answer,
            model_answer=model_answer,
        )
        judge_failed = result.reasoning.startswith("[JUDGE FAILED]")
        final = kw_score if judge_failed and use_keyword_fallback else result.score

        return GenerationScore(
            metric="context_recall",
            llm_score=result.score,
            keyword_score=kw_score,
            final_score=round(final, 3),
            reasoning=result.reasoning,
            judge_failed=judge_failed,
            judge_latency_s=result.latency_s,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Context recall judge error: %s", exc)
        return GenerationScore(
            metric="context_recall",
            llm_score=kw_score,
            keyword_score=kw_score,
            final_score=kw_score,
            reasoning=f"[JUDGE EXCEPTION] {exc} — keyword fallback used.",
            judge_failed=True,
        )


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


async def compute_all_generation_metrics(
    query: str,
    answer: str,
    retrieved_context: str,
    reference_answer: str,
    judge: Optional[LLMJudge] = None,
) -> Dict[str, GenerationScore]:
    """
    Compute all three generation metrics concurrently for a single query.

    Args:
        query:             User's legal question.
        answer:            Chatbot's generated response.
        retrieved_context: Full RAG / Indian Kanoon context fed to the LLM.
        reference_answer:  Gold-standard answer for context recall evaluation.
        judge:             Optional shared :class:`~app.metrics.llm_judge.LLMJudge`.

    Returns:
        Dict keyed by metric name::

            {
                "faithfulness":     GenerationScore,
                "answer_relevance": GenerationScore,
                "context_recall":   GenerationScore,
            }
    """
    j = judge or get_judge()

    faith_task = compute_faithfulness(
        answer=answer, retrieved_context=retrieved_context, judge=j
    )
    relevance_task = compute_answer_relevance(query=query, answer=answer, judge=j)
    recall_task = compute_context_recall(
        reference_answer=reference_answer, model_answer=answer, judge=j
    )

    faith, relevance, recall = await asyncio.gather(
        faith_task,
        relevance_task,
        recall_task,
        return_exceptions=False,
    )

    return {
        "faithfulness": faith,
        "answer_relevance": relevance,
        "context_recall": recall,
    }


def aggregate_generation_scores(
    records: Sequence[Dict[str, GenerationScore]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-query generation scores into dataset-level statistics.

    Args:
        records: List of dicts, each returned by
                 :func:`compute_all_generation_metrics`.

    Returns:
        Nested dict::

            {
                "faithfulness": {
                    "mean":  float,
                    "min":   float,
                    "max":   float,
                    "count": int,
                    "judge_failures": int,
                },
                "answer_relevance": { ... },
                "context_recall":   { ... },
            }
    """
    metrics = ["faithfulness", "answer_relevance", "context_recall"]
    result: Dict[str, Dict[str, float]] = {}

    for metric in metrics:
        scores = [r[metric].final_score for r in records if metric in r]
        failures = sum(1 for r in records if metric in r and r[metric].judge_failed)

        if not scores:
            result[metric] = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "judge_failures": 0,
            }
            continue

        result[metric] = {
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "count": len(scores),
            "judge_failures": failures,
        }

    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio

    _query = "Can an FIR be quashed by the High Court?"
    _context = (
        "Section 482 CrPC grants the High Court inherent powers to quash an FIR "
        "to prevent abuse of process. In Bhajan Lal (1992) the Supreme Court listed "
        "exhaustive grounds for quashing."
    )
    _answer = (
        "Yes, an FIR can be quashed by the High Court under Section 482 CrPC "
        "if the allegations do not disclose any offence or are ex facie false, "
        "as held in Bhajan Lal v. State of Haryana."
    )
    _reference = (
        "Section 482 CrPC — inherent powers of High Court. Grounds from Bhajan Lal: "
        "no offence disclosed, ex facie false FIR, abuse of process, civil dispute "
        "dressed as criminal."
    )

    async def _demo() -> None:
        scores = await compute_all_generation_metrics(
            query=_query,
            answer=_answer,
            retrieved_context=_context,
            reference_answer=_reference,
        )
        for name, s in scores.items():
            print(f"  {name:<20} final={s.final_score:.2f}  [{s.label}]")
            print(f"    reasoning: {s.reasoning}")

    _asyncio.run(_demo())
