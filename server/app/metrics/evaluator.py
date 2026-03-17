"""
MetricsEvaluator -- Main Orchestrator
======================================
Runs all 9 evaluation metrics against the legal chatbot RAG pipeline.

Metrics
-------
  RETRIEVAL
  * Hit Rate @ 1/3/5  -- % of queries where a ground-truth IPC section is in top-k
  * MRR               -- Mean Reciprocal Rank of the first relevant section
  * Context Precision -- Fraction of retrieved context that is relevant (LLM judge)

  GENERATION (RAG Triad)
  * Faithfulness      -- % of answer claims grounded in retrieved context
  * Answer Relevance  -- Does the answer address the question?
  * Context Recall    -- Does the answer cover key facts from the reference?

  ENGINEERING & COST
  * Latency stats     -- min / p50 / p95 / p99 / max / stdev (seconds)
  * Cost per 1k       -- Estimated USD at four pricing tiers
  * Token Efficiency  -- Output tokens / total tokens ratio

Architecture (two-pass)
-----------------------
  Pass 1 -- Pre-computed chatbot results from test_chatbot.py
            (answer, intent, latency already captured)
  Pass 2 -- Direct RAG re-retrieval inside the evaluator
            (ranked sections + raw context for retrieval / judge metrics)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.metrics.engineering_metrics import (
    aggregate_token_efficiency,
    build_token_record,
    compute_latency_stats,
    compute_token_efficiency,
    count_tokens_approx,
    estimate_cost_all_tiers,
)
from app.metrics.generation_metrics import (
    GenerationScore,
    aggregate_generation_scores,
    compute_all_generation_metrics,
)
from app.metrics.ground_truth import (
    GROUND_TRUTH,
    GroundTruthEntry,
    get_entry_by_query,
)
from app.metrics.llm_judge import LLMJudge
from app.metrics.retrieval_metrics import (
    RetrievalSample,
    compute_all_retrieval_metrics,
    compute_hit_rate,
    compute_mrr_single,
    print_retrieval_report,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Per-query result container
# =============================================================================


@dataclass
class EvalResult:
    """All evaluation data for a single query."""

    # Identity
    query_id: int
    query: str
    domain: str
    intent: str

    # Chatbot output
    answer: str
    latency_s: float

    # Retrieved context (from direct RAG pass)
    retrieved_sections: List[str]
    retrieved_context: str

    # Retrieval metrics
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    reciprocal_rank: float = 0.0
    context_precision: float = 0.0
    context_precision_reasoning: str = ""

    # Generation metrics
    faithfulness: float = 0.0
    faithfulness_reasoning: str = ""
    faithfulness_from_llm: bool = False

    answer_relevance: float = 0.0
    answer_relevance_reasoning: str = ""
    answer_relevance_from_llm: bool = False

    context_recall: float = 0.0
    context_recall_reasoning: str = ""
    context_recall_from_llm: bool = False

    # Engineering metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_efficiency_score: float = 0.0
    output_ratio: float = 0.0

    # Meta
    judge_failures: int = 0
    error: Optional[str] = None

    @property
    def rag_triad_mean(self) -> float:
        """Mean of the three RAG-triad generation metrics."""
        return round(
            (self.faithfulness + self.answer_relevance + self.context_recall) / 3, 4
        )

    @property
    def overall_score(self) -> float:
        """
        Weighted composite score.
          Faithfulness 20% + Answer Relevance 20% + Context Recall 10%
          + MRR 25% + Context Precision 15% + Token Efficiency 10%
        """
        return round(
            self.faithfulness * 0.20
            + self.answer_relevance * 0.20
            + self.context_recall * 0.10
            + self.reciprocal_rank * 0.25
            + self.context_precision * 0.15
            + self.token_efficiency_score * 0.10,
            4,
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flat dict for CSV export. Truncates reasoning; drops raw context."""
        d = asdict(self)
        d["rag_triad_mean"] = self.rag_triad_mean
        d["overall_score"] = self.overall_score
        for key in list(d.keys()):
            if key.endswith("_reasoning") and isinstance(d[key], str):
                d[key] = d[key][:150].replace("\n", " ")
        if isinstance(d.get("retrieved_sections"), list):
            d["retrieved_sections"] = "|".join(d["retrieved_sections"])
        d.pop("retrieved_context", None)
        return d


# =============================================================================
# Aggregate report container
# =============================================================================


@dataclass
class AggregateReport:
    """Dataset-level aggregated statistics."""

    total_queries: int
    errors: int
    judge_failures: int

    # Retrieval
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float
    context_precision_mean: float

    # Generation
    faithfulness_mean: float
    answer_relevance_mean: float
    context_recall_mean: float
    rag_triad_mean: float

    # Engineering
    latency: Dict[str, float]
    cost_estimates: Dict[str, Dict]
    token_efficiency: Dict[str, float]

    # Domain breakdown
    domain_scores: Dict[str, Dict[str, float]]

    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# MetricsEvaluator
# =============================================================================


class MetricsEvaluator:
    """
    Orchestrates the full 9-metric evaluation suite for the legal chatbot.

    Parameters
    ----------
    use_llm_judge : bool
        True  -> local Ollama LLM is used for all judge-based metrics.
        False -> keyword heuristics only (offline / fast mode).
    rag_k : int
        Number of IPC sections to retrieve in the direct RAG pass.
    max_concurrent_judge_calls : int
        Semaphore cap to avoid flooding the local Ollama server.
    """

    def __init__(
        self,
        use_llm_judge: bool = True,
        rag_k: int = 5,
        max_concurrent_judge_calls: int = 3,
    ) -> None:
        self.use_llm_judge = use_llm_judge
        self.rag_k = rag_k
        self._judge: Optional[LLMJudge] = LLMJudge() if use_llm_judge else None
        self._judge_sem = asyncio.Semaphore(max_concurrent_judge_calls)

    # -------------------------------------------------------------------------
    # Pass 2: direct RAG instrumentation
    # -------------------------------------------------------------------------

    async def _get_rag_context(
        self, query: str, gt_entry: GroundTruthEntry
    ) -> Tuple[List[str], str]:
        """
        Re-run retrieval for *query* using domain-aware strategy.

        Crime RAG (IPC FAISS) is invoked ONLY when the ground truth expects
        IPC sections.  Indian Kanoon uses the ground truth domain to pick
        the most relevant ``context_type``.

        Returns
        -------
        (ranked_section_numbers, concatenated_context_text)
        """
        sections: List[str] = []
        ctx_parts: List[str] = []

        has_ipc_ground_truth = bool(gt_entry.get("relevant_ipc_sections"))

        # ── 1. Crime RAG (IPC FAISS) — only for criminal/IPC queries ──
        if has_ipc_ground_truth:
            try:
                from app.tools.crime_rag import extract_crime_features, get_rag_system

                rag = get_rag_system()
                await rag.initialize()

                if rag.initialized:
                    features = extract_crime_features(query)
                    rag_result = await rag.retrieve_sections(
                        query, crime_type="general", features=features, k=self.rag_k
                    )

                    sections = [m.section for m in rag_result.ipc_sections]
                    ctx_parts.extend(
                        f"IPC Section {m.section} -- {m.title}\n"
                        f"Punishment: {m.punishment}\n"
                        f"{m.definition[:400]}"
                        for m in rag_result.ipc_sections
                    )
            except Exception as exc:
                logger.warning("Crime RAG retrieval failed during evaluation: %s", exc)

        # ── 2. Indian Kanoon (case law & statutes API) ────────────────
        # Map ground-truth domain → Indian Kanoon context_type
        domain = gt_entry.get("domain", "unknown")
        ik_context_map = {
            "constitutional": "constitution",
            "criminal": "ipc",
            "criminal_procedure": "crpc",
            "contract_law": "statute",
            "property_law": "statute",
            "family_law": "statute",
            "evidence_law": "statute",
            "technology_law": "general",
        }
        ik_context_type = ik_context_map.get(domain, "general")

        try:
            from app.tools.indian_kanoon import get_indian_kanoon_tool

            ik_tool = get_indian_kanoon_tool()
            await ik_tool.initialize()
            result = await ik_tool.answer_legal_query(
                query, context_type=ik_context_type
            )
            ik_results = result.get("results", [])

            for doc in ik_results[:5]:
                excerpt = getattr(doc, "excerpt", "") or ""
                title = getattr(doc, "title", "") or ""
                if excerpt or title:
                    ctx_parts.append(
                        f"{title}\n{excerpt[:500]}"
                    )
        except Exception as exc:
            logger.warning("Indian Kanoon retrieval failed during evaluation: %s", exc)

        return sections, "\n\n".join(ctx_parts)

    # -------------------------------------------------------------------------
    # Single-query evaluation
    # -------------------------------------------------------------------------

    async def _evaluate_one(
        self,
        query_id: int,
        chatbot_result: Dict[str, Any],
        gt_entry: GroundTruthEntry,
    ) -> EvalResult:
        """Compute all 9 metrics for a single query."""

        query = chatbot_result.get("query", gt_entry["query"])
        answer = chatbot_result.get("answer", "")
        intent = chatbot_result.get("intent", "unknown")
        latency = float(chatbot_result.get("response_time_s", 0.0))
        error: Optional[str] = None

        if isinstance(answer, str) and answer.startswith("ERROR"):
            error, answer = answer, ""

        # ---- Pass 2: RAG retrieval -------------------------------------------
        retrieved_sections, retrieved_context = await self._get_rag_context(
            query, gt_entry
        )

        # ---- Retrieval metrics (pure, no LLM) --------------------------------
        relevant_secs = gt_entry["relevant_ipc_sections"]
        hr1 = compute_hit_rate(retrieved_sections, relevant_secs, k=1)
        hr3 = compute_hit_rate(retrieved_sections, relevant_secs, k=3)
        hr5 = compute_hit_rate(retrieved_sections, relevant_secs, k=5)
        rr = compute_mrr_single(retrieved_sections, relevant_secs)

        # ---- Engineering metrics (pure, no LLM) ------------------------------
        token_record = build_token_record(
            prompt=query, context=retrieved_context, answer=answer
        )
        # Add ~500-token system-prompt overhead approximation
        token_record["input_tokens"] = max(1, token_record["input_tokens"] + 500)
        eff = compute_token_efficiency(
            answer=answer,
            input_tokens=token_record["input_tokens"],
            output_tokens=token_record["output_tokens"],
        )

        # ---- LLM judge metrics (context precision + RAG triad) ---------------
        judge_failures = 0
        cp_precision = 0.0
        gen_scores: Dict[str, GenerationScore] = {}

        async def _guarded(coro):
            async with self._judge_sem:
                return await coro

        if answer:
            try:
                from app.metrics.retrieval_metrics import (
                    compute_context_precision_async,
                )

                cp_raw, gen_scores = await asyncio.gather(
                    _guarded(
                        compute_context_precision_async(
                            query=query,
                            context=retrieved_context,
                            keywords=gt_entry["relevant_keywords"],
                            judge=self._judge,
                        )
                    ),
                    _guarded(
                        compute_all_generation_metrics(
                            query=query,
                            answer=answer,
                            retrieved_context=retrieved_context,
                            reference_answer=gt_entry["reference_answer"],
                            judge=self._judge,
                        )
                    ),
                    return_exceptions=False,
                )
                cp_precision = float(cp_raw)

            except Exception as exc:
                logger.warning("Metrics failed for query %d: %s", query_id, exc)
                judge_failures += 4
        else:
            judge_failures += 4  # error / empty answer

        faith_s = gen_scores.get("faithfulness")
        rel_s = gen_scores.get("answer_relevance")
        recall_s = gen_scores.get("context_recall")

        if faith_s and faith_s.judge_failed:
            judge_failures += 1
        if rel_s and rel_s.judge_failed:
            judge_failures += 1
        if recall_s and recall_s.judge_failed:
            judge_failures += 1

        return EvalResult(
            query_id=query_id,
            query=query,
            domain=gt_entry["domain"],
            intent=intent,
            answer=answer,
            latency_s=latency,
            retrieved_sections=retrieved_sections,
            retrieved_context=retrieved_context,
            hit_rate_at_1=hr1,
            hit_rate_at_3=hr3,
            hit_rate_at_5=hr5,
            reciprocal_rank=rr,
            context_precision=round(cp_precision, 4),
            context_precision_reasoning="",
            faithfulness=faith_s.final_score if faith_s else 0.0,
            faithfulness_reasoning=(faith_s.reasoning if faith_s else "Not evaluated."),
            faithfulness_from_llm=not (faith_s.judge_failed if faith_s else True),
            answer_relevance=rel_s.final_score if rel_s else 0.0,
            answer_relevance_reasoning=(rel_s.reasoning if rel_s else "Not evaluated."),
            answer_relevance_from_llm=not (rel_s.judge_failed if rel_s else True),
            context_recall=recall_s.final_score if recall_s else 0.0,
            context_recall_reasoning=(
                recall_s.reasoning if recall_s else "Not evaluated."
            ),
            context_recall_from_llm=not (recall_s.judge_failed if recall_s else True),
            input_tokens=eff["input_tokens"],
            output_tokens=eff["output_tokens"],
            total_tokens=eff["total_tokens"],
            token_efficiency_score=eff["efficiency_score"],
            output_ratio=eff["output_ratio"],
            judge_failures=judge_failures,
            error=error,
        )

    # -------------------------------------------------------------------------
    # Public: run
    # -------------------------------------------------------------------------

    async def run(self, chatbot_results: List[Dict[str, Any]]) -> List[EvalResult]:
        """
        Evaluate all chatbot results concurrently.

        Parameters
        ----------
        chatbot_results : list[dict]
            Output of test_chatbot.py::run_evaluation().
            Required keys: query, answer, intent, response_time_s.

        Returns
        -------
        list[EvalResult] -- one per query, same order as input.
        """
        tasks = []
        for i, cr in enumerate(chatbot_results, start=1):
            query = cr.get("query", "")
            gt = get_entry_by_query(query)
            if gt is None:
                gt = GroundTruthEntry(
                    query=query,
                    relevant_ipc_sections=[],
                    relevant_keywords=[],
                    expected_acts=[],
                    reference_answer="",
                    domain="unknown",
                )
            tasks.append(self._evaluate_one(i, cr, gt))

        mode = "on" if self.use_llm_judge else "off"
        print(
            f"\n[MetricsEvaluator] Evaluating {len(tasks)} queries "
            f"(llm_judge={mode}) ..."
        )
        t0 = time.perf_counter()
        results: List[EvalResult] = list(
            await asyncio.gather(*tasks, return_exceptions=False)
        )
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"[MetricsEvaluator] Done in {elapsed}s.\n")
        return results

    # -------------------------------------------------------------------------
    # Public: aggregate
    # -------------------------------------------------------------------------

    def aggregate(self, results: List[EvalResult]) -> AggregateReport:
        """Compute dataset-level statistics from a list of EvalResult."""
        n = len(results)
        if n == 0:
            raise ValueError("Cannot aggregate an empty results list.")

        errors = sum(1 for r in results if r.error)

        # Retrieval
        hr1 = round(sum(r.hit_rate_at_1 for r in results) / n, 4)
        hr3 = round(sum(r.hit_rate_at_3 for r in results) / n, 4)
        hr5 = round(sum(r.hit_rate_at_5 for r in results) / n, 4)
        cp_mean = round(sum(r.context_precision for r in results) / n, 4)

        gt_queries_with_secs = {
            gt["query"] for gt in GROUND_TRUTH if gt["relevant_ipc_sections"]
        }
        rrs = [r.reciprocal_rank for r in results if r.query in gt_queries_with_secs]
        mrr = round(sum(rrs) / len(rrs), 4) if rrs else 0.0

        # Generation
        faith_mean = round(sum(r.faithfulness for r in results) / n, 4)
        rel_mean = round(sum(r.answer_relevance for r in results) / n, 4)
        recall_mean = round(sum(r.context_recall for r in results) / n, 4)
        rag_triad = round((faith_mean + rel_mean + recall_mean) / 3, 4)

        # Engineering
        latency_stats = compute_latency_stats([r.latency_s for r in results])
        token_records = [
            {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens}
            for r in results
        ]
        cost_estimates = estimate_cost_all_tiers(token_records)
        eff_records = [
            {
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "output_ratio": r.output_ratio,
                "efficiency_score": r.token_efficiency_score,
            }
            for r in results
        ]
        token_eff = aggregate_token_efficiency(eff_records)

        # Domain breakdown
        domain_groups: Dict[str, List[EvalResult]] = {}
        for r in results:
            domain_groups.setdefault(r.domain, []).append(r)

        domain_scores: Dict[str, Dict[str, float]] = {}
        for domain, group in domain_groups.items():
            m = len(group)
            domain_scores[domain] = {
                "count": float(m),
                "faithfulness": round(sum(x.faithfulness for x in group) / m, 4),
                "answer_relevance": round(
                    sum(x.answer_relevance for x in group) / m, 4
                ),
                "context_recall": round(sum(x.context_recall for x in group) / m, 4),
                "context_precision": round(
                    sum(x.context_precision for x in group) / m, 4
                ),
                "mrr": round(sum(x.reciprocal_rank for x in group) / m, 4),
                "latency_mean": round(sum(x.latency_s for x in group) / m, 2),
            }

        return AggregateReport(
            total_queries=n,
            errors=errors,
            judge_failures=sum(r.judge_failures for r in results),
            hit_rate_at_1=hr1,
            hit_rate_at_3=hr3,
            hit_rate_at_5=hr5,
            mrr=mrr,
            context_precision_mean=cp_mean,
            faithfulness_mean=faith_mean,
            answer_relevance_mean=rel_mean,
            context_recall_mean=recall_mean,
            rag_triad_mean=rag_triad,
            latency=latency_stats,
            cost_estimates=cost_estimates,
            token_efficiency=token_eff,
            domain_scores=domain_scores,
        )

    # -------------------------------------------------------------------------
    # Public: print_report
    # -------------------------------------------------------------------------

    def print_report(
        self,
        results: List[EvalResult],
        report: Optional[AggregateReport] = None,
    ) -> None:
        """Print a rich textual evaluation report to stdout."""
        if report is None:
            report = self.aggregate(results)

        W = 70
        sep = "=" * W
        thn = "-" * W

        def _bar(score: float, w: int = 20) -> str:
            filled = int(round(score * w))
            return "[" + "#" * filled + "." * (w - filled) + f"] {score:.3f}"

        def _lbl(s: float) -> str:
            if s >= 0.85:
                return "EXCELLENT"
            if s >= 0.70:
                return "GOOD"
            if s >= 0.50:
                return "FAIR"
            if s >= 0.30:
                return "POOR"
            return "VERY POOR"

        print(f"\n{sep}")
        print("  LEGAL CHATBOT -- FULL METRICS REPORT")
        print(f"  Evaluated     : {report.evaluated_at}")
        print(
            f"  Queries       : {report.total_queries}  |  "
            f"Errors: {report.errors}  |  "
            f"Judge failures: {report.judge_failures}"
        )
        jmode = (
            "ON (Ollama LLM-as-Judge)"
            if self.use_llm_judge
            else "OFF (keyword heuristics only)"
        )
        print(f"  LLM Judge     : {jmode}")
        print(sep)

        # [1] Retrieval --------------------------------------------------------
        n_gt = sum(1 for g in GROUND_TRUTH if g["relevant_ipc_sections"])
        print(
            f"\n  [1] RETRIEVAL METRICS"
            f"\n      (IPC ground truth: {n_gt}/{len(GROUND_TRUTH)} queries)"
        )
        print(thn)
        print(f"  Hit Rate @ 1          {_bar(report.hit_rate_at_1)}")
        print(f"  Hit Rate @ 3          {_bar(report.hit_rate_at_3)}")
        print(f"  Hit Rate @ 5          {_bar(report.hit_rate_at_5)}")
        print(f"  Mean Reciprocal Rank  {_bar(report.mrr)}")
        print(f"  Context Precision     {_bar(report.context_precision_mean)}")

        # [2] Generation -------------------------------------------------------
        print("\n  [2] GENERATION METRICS  (RAG Triad)")
        print(thn)
        print(
            f"  Faithfulness          {_bar(report.faithfulness_mean)}"
            f"  {_lbl(report.faithfulness_mean)}"
        )
        print(
            f"  Answer Relevance      {_bar(report.answer_relevance_mean)}"
            f"  {_lbl(report.answer_relevance_mean)}"
        )
        print(
            f"  Context Recall        {_bar(report.context_recall_mean)}"
            f"  {_lbl(report.context_recall_mean)}"
        )
        print(thn)
        print(
            f"  RAG Triad Mean        {_bar(report.rag_triad_mean)}"
            f"  {_lbl(report.rag_triad_mean)}"
        )

        # [3] Engineering -------------------------------------------------------
        lat = report.latency
        print("\n  [3] ENGINEERING & COST METRICS")
        print(thn)
        print("  Latency (seconds per request)")
        print(f"    min    = {lat['min']:.2f}s")
        print(f"    mean   = {lat['mean']:.2f}s")
        print(f"    p50    = {lat['median']:.2f}s")
        print(f"    p95    = {lat['p95']:.2f}s")
        print(f"    p99    = {lat['p99']:.2f}s")
        print(f"    max    = {lat['max']:.2f}s")
        print(f"    stdev  = {lat['stdev']:.2f}s")

        te = report.token_efficiency
        print("\n  Token Usage Efficiency")
        print(f"    avg input tokens    = {te['avg_input_tokens']:.0f}")
        print(f"    avg output tokens   = {te['avg_output_tokens']:.0f}")
        print(f"    avg total tokens    = {te['avg_total_tokens']:.0f}")
        print(f"    output / input      = {te['avg_output_ratio']:.3f}")
        print(
            f"    efficiency score    = {te['avg_efficiency_score']:.3f}"
            "  (ideal: 0.10 - 0.40)"
        )

        print("\n  Cost per 1,000 Queries  (USD estimates)")
        print(f"  {'Model / Tier':<52}  {'$/1k':>8}  {'$/10k':>8}")
        print(f"  {'':-<52}  {'':-<8}  {'':-<8}")
        for tier, info in report.cost_estimates.items():
            desc = info.get("description", tier)
            per1k = info.get("cost_per_1k_usd", 0.0)
            per10k = info.get("cost_per_10k_usd", 0.0)
            print(f"  {desc:<52}  ${per1k:>7.4f}  ${per10k:>7.3f}")

        # [4] Domain breakdown -------------------------------------------------
        print("\n  [4] DOMAIN BREAKDOWN")
        print(thn)
        print(
            f"  {'Domain':<22} {'N':>3}  {'Faith':>7}  {'Relev':>7}"
            f"  {'Recall':>7}  {'MRR':>7}  {'Lat(s)':>7}"
        )
        print(f"  {'':-<22} {'':-<3}  {'':-<7}  {'':-<7}  {'':-<7}  {'':-<7}  {'':-<7}")
        for domain, ds in sorted(report.domain_scores.items()):
            print(
                f"  {domain:<22} {int(ds['count']):>3}  "
                f"{ds['faithfulness']:>7.3f}  "
                f"{ds['answer_relevance']:>7.3f}  "
                f"{ds['context_recall']:>7.3f}  "
                f"{ds['mrr']:>7.3f}  "
                f"{ds['latency_mean']:>7.1f}"
            )

        # [5] Per-query detail -------------------------------------------------
        print("\n  [5] PER-QUERY DETAIL")
        print(thn)
        print(
            f"  {'#':>3}  {'Domain':<18}  {'Faith':>7}  {'Relev':>7}"
            f"  {'Recall':>7}  {'Prec':>7}  {'RR':>6}  {'Lat':>6}"
        )
        print(
            f"  {'':-<3}  {'':-<18}  {'':-<7}  {'':-<7}"
            f"  {'':-<7}  {'':-<7}  {'':-<6}  {'':-<6}"
        )
        for r in results:
            flag = " ERR" if r.error else ""
            print(
                f"  {r.query_id:>3}  {r.domain:<18}  "
                f"{r.faithfulness:>7.3f}  "
                f"{r.answer_relevance:>7.3f}  "
                f"{r.context_recall:>7.3f}  "
                f"{r.context_precision:>7.3f}  "
                f"{r.reciprocal_rank:>6.3f}  "
                f"{r.latency_s:>6.1f}{flag}"
            )

        print(f"\n{sep}\n")

    # -------------------------------------------------------------------------
    # Public: save_csv
    # -------------------------------------------------------------------------

    def save_csv(self, results: List[EvalResult], path: str) -> None:
        """Save per-query results to a CSV file."""
        if not results:
            print("[MetricsEvaluator] No results to save.")
            return
        rows = [r.to_flat_dict() for r in results]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[MetricsEvaluator] CSV saved  -> {path}")

    # -------------------------------------------------------------------------
    # Public: save_json
    # -------------------------------------------------------------------------

    def save_json(self, results: List[EvalResult], path: str) -> None:
        """Save the full report (per-query + aggregate) to JSON."""
        report = self.aggregate(results)
        payload = {
            "meta": {
                "evaluated_at": report.evaluated_at,
                "total_queries": report.total_queries,
                "errors": report.errors,
                "judge_failures": report.judge_failures,
                "llm_judge": self.use_llm_judge,
                "rag_k": self.rag_k,
            },
            "aggregate": {
                "retrieval": {
                    "hit_rate_at_1": report.hit_rate_at_1,
                    "hit_rate_at_3": report.hit_rate_at_3,
                    "hit_rate_at_5": report.hit_rate_at_5,
                    "mrr": report.mrr,
                    "context_precision_mean": report.context_precision_mean,
                },
                "generation": {
                    "faithfulness_mean": report.faithfulness_mean,
                    "answer_relevance_mean": report.answer_relevance_mean,
                    "context_recall_mean": report.context_recall_mean,
                    "rag_triad_mean": report.rag_triad_mean,
                },
                "engineering": {
                    "latency": report.latency,
                    "cost_estimates": report.cost_estimates,
                    "token_efficiency": report.token_efficiency,
                },
                "domain_scores": report.domain_scores,
            },
            "per_query": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "domain": r.domain,
                    "intent": r.intent,
                    "latency_s": r.latency_s,
                    "retrieved_sections": r.retrieved_sections,
                    "metrics": {
                        "hit_rate_at_1": r.hit_rate_at_1,
                        "hit_rate_at_3": r.hit_rate_at_3,
                        "hit_rate_at_5": r.hit_rate_at_5,
                        "reciprocal_rank": r.reciprocal_rank,
                        "context_precision": r.context_precision,
                        "faithfulness": r.faithfulness,
                        "answer_relevance": r.answer_relevance,
                        "context_recall": r.context_recall,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "total_tokens": r.total_tokens,
                        "token_efficiency": r.token_efficiency_score,
                        "output_ratio": r.output_ratio,
                        "rag_triad_mean": r.rag_triad_mean,
                        "overall_score": r.overall_score,
                    },
                    "reasoning": {
                        "faithfulness": r.faithfulness_reasoning,
                        "answer_relevance": r.answer_relevance_reasoning,
                        "context_recall": r.context_recall_reasoning,
                    },
                    "judge_failures": r.judge_failures,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[MetricsEvaluator] JSON saved -> {path}")
