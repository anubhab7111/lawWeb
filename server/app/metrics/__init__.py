"""
Legal Chatbot Metrics Package
=============================
Implements the full evaluation suite for the RAG-based legal chatbot:

  Retrieval Metrics
  -----------------
  • Hit Rate @ k   — % of queries where a relevant document is in the top-k results
  • MRR            — Mean Reciprocal Rank of the first relevant result
  • Context Precision — Fraction of retrieved context that is actually relevant

  Generation Metrics  (RAG Triad)
  --------------------------------
  • Faithfulness      — How grounded the answer is in the retrieved context
  • Answer Relevance  — Does the answer address the user's question?
  • Context Recall    — Does the answer cover the key facts in the retrieved context?

  Engineering & Cost Metrics
  --------------------------
  • Latency Stats           — p50 / p95 / mean seconds per request
  • Cost per 1 k Queries    — Estimated LLM spend at cloud API rates
  • Token Efficiency        — Ratio of answer tokens to total tokens processed

Recommended usage
-----------------
  from app.metrics import MetricsEvaluator

  evaluator = MetricsEvaluator()
  report    = await evaluator.run(results)   # list[EvalResult]
  evaluator.print_report(report)
  evaluator.save_report(report, "metrics_report.csv")
"""

from app.metrics.engineering_metrics import (
    compute_latency_stats,
    compute_token_efficiency,
    count_tokens_approx,
    estimate_cost_per_1k,
)
from app.metrics.evaluator import EvalResult, MetricsEvaluator
from app.metrics.generation_metrics import (
    compute_answer_relevance,
    compute_context_recall,
    compute_faithfulness,
)
from app.metrics.ground_truth import GROUND_TRUTH, GroundTruthEntry
from app.metrics.llm_judge import LLMJudge
from app.metrics.retrieval_metrics import (
    compute_context_precision,
    compute_hit_rate,
    compute_mrr,
)

__all__ = [
    # Ground truth
    "GROUND_TRUTH",
    "GroundTruthEntry",
    # LLM judge
    "LLMJudge",
    # Retrieval
    "compute_hit_rate",
    "compute_mrr",
    "compute_context_precision",
    # Generation
    "compute_faithfulness",
    "compute_answer_relevance",
    "compute_context_recall",
    # Engineering
    "compute_latency_stats",
    "estimate_cost_per_1k",
    "compute_token_efficiency",
    "count_tokens_approx",
    # Orchestrator
    "MetricsEvaluator",
    "EvalResult",
]
