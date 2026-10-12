"""Legal Chatbot Metrics Package ============================= Implements the full evaluation suite for the RAG-based legal chatbot: Retrieval Metrics ----------------- • Hit Rate..."""

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

    "GROUND_TRUTH",
    "GroundTruthEntry",

    "LLMJudge",

    "compute_hit_rate",
    "compute_mrr",
    "compute_context_precision",

    "compute_faithfulness",
    "compute_answer_relevance",
    "compute_context_recall",

    "compute_latency_stats",
    "estimate_cost_per_1k",
    "compute_token_efficiency",
    "count_tokens_approx",

    "MetricsEvaluator",
    "EvalResult",
]
