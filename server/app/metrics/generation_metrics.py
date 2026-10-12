"""Generation Metrics (The RAG Triad) ==================================== Implements the three core generation-quality metrics for a RAG pipeline: 1."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional

from app.metrics.llm_judge import JudgeScore, LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class GenerationScore:
    """Enriched result for a single generation metric evaluation."""

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


def _keyword_faithfulness(answer: str, context: str) -> float:
    """Heuristic faithfulness score."""
    if not answer.strip() or not context.strip():
        return 0.0

    answer_lower = answer.lower()
    context_lower = context.lower()

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

    hallucination_signals = [
        "section 243b",
        "negligence act, 1872",
        "indian negligence act",
        "section 66 of the constitution",
    ]
    hallucination_hits = sum(1 for p in hallucination_signals if p in answer_lower)

    answer_words = set(re.findall(r"\b[a-z]{5,}\b", answer_lower))
    context_words = set(re.findall(r"\b[a-z]{5,}\b", context_lower))
    overlap = len(answer_words & context_words)
    overlap_ratio = min(overlap / max(len(answer_words), 1), 1.0)

    base = overlap_ratio * 0.7

    base = min(base + min(grounding_hits, 5) * 0.02, 1.0)

    penalty = hallucination_hits * 0.15
    return round(max(0.0, min(base - penalty, 1.0)), 3)


def _keyword_answer_relevance(query: str, answer: str) -> float:
    """Heuristic answer relevance score."""
    if not answer.strip():
        return 0.0
    if answer.startswith("ERROR") or "I'm having trouble" in answer:
        return 0.1

    query_lower = query.lower()
    answer_lower = answer.lower()

    query_terms = set(re.findall(r"\b[a-z]{4,}\b", query_lower))
    if not query_terms:
        return 0.5

    answer_terms = set(re.findall(r"\b[a-z]{4,}\b", answer_lower))
    matched = query_terms & answer_terms
    term_coverage = len(matched) / len(query_terms)

    length_score = min(len(answer) / 500, 1.0)

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
    """Heuristic context recall: keyword overlap between reference and model answer."""
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


async def compute_faithfulness(
    answer: str,
    retrieved_context: str,
    judge: Optional[LLMJudge] = None,
    *,
    use_keyword_fallback: bool = True,
) -> GenerationScore:
    """Faithfulness (Groundedness)."""
    kw_score = (
        _keyword_faithfulness(answer, retrieved_context)
        if use_keyword_fallback
        else 0.5
    )
    if judge is None:
        return GenerationScore(
            metric="faithfulness",
            llm_score=kw_score,
            keyword_score=kw_score,
            final_score=round(kw_score, 3),
            reasoning="LLM judge disabled — keyword heuristic used.",
            judge_failed=True,
        )

    try:
        result: JudgeScore = await judge.faithfulness(
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
    """Answer Relevance."""
    kw_score = _keyword_answer_relevance(query, answer) if use_keyword_fallback else 0.5
    if judge is None:
        return GenerationScore(
            metric="answer_relevance",
            llm_score=kw_score,
            keyword_score=kw_score,
            final_score=round(kw_score, 3),
            reasoning="LLM judge disabled — keyword heuristic used.",
            judge_failed=True,
        )

    try:
        result: JudgeScore = await judge.answer_relevance(
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
    """Context Recall."""
    kw_score = (
        _keyword_context_recall(reference_answer, model_answer)
        if use_keyword_fallback
        else 0.5
    )
    if judge is None:
        return GenerationScore(
            metric="context_recall",
            llm_score=kw_score,
            keyword_score=kw_score,
            final_score=round(kw_score, 3),
            reasoning="LLM judge disabled — keyword heuristic used.",
            judge_failed=True,
        )

    try:
        result: JudgeScore = await judge.context_recall(
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


async def compute_all_generation_metrics(
    query: str,
    answer: str,
    retrieved_context: str,
    reference_answer: str,
    judge: Optional[LLMJudge] = None,
) -> Dict[str, GenerationScore]:
    """Compute all three generation metrics concurrently for a single query."""
    faith_task = compute_faithfulness(
        answer=answer, retrieved_context=retrieved_context, judge=judge
    )
    relevance_task = compute_answer_relevance(query=query, answer=answer, judge=judge)
    recall_task = compute_context_recall(
        reference_answer=reference_answer, model_answer=answer, judge=judge
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
