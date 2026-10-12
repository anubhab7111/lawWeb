"""Engineering & Cost Metrics ========================== Tracks the operational health and economic efficiency of the RAG pipeline."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def count_tokens_approx(text: str) -> int:
    """Approximate the number of LLM tokens in *text*."""
    if not text or not text.strip():
        return 0

    cleaned = _WHITESPACE_RE.sub(" ", text.strip())
    word_count = len(cleaned.split())


    token_count = math.ceil(word_count * 1.333 * 1.10)
    return max(1, token_count)


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Compute a percentile value from a **sorted** list using linear interpolation (same algorithm as numpy.percentile with interpolation='linear')."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]

    idx = (pct / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    """Compute a full latency distribution from a list of per-request timings."""
    if not latencies:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "stdev": 0.0,
        }

    values = sorted(float(v) for v in latencies)
    n = len(values)
    mean = sum(values) / n


    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0

    return {
        "count": n,
        "min": round(values[0], 3),
        "mean": round(mean, 3),
        "median": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(values[-1], 3),
        "stdev": round(stdev, 3),
    }


PRICING_TABLE: Dict[str, Dict[str, float]] = {


    "ollama_local": {
        "input_per_1k_tokens": 0.00010,
        "output_per_1k_tokens": 0.00010,
        "description": "Self-hosted Ollama (Mistral-7B / LLaMA-3-8B)",
    },

    "cloud_small": {
        "input_per_1k_tokens": 0.00020,
        "output_per_1k_tokens": 0.00020,
        "description": "Cloud small (Mistral-7B / LLaMA-3-8B via API)",
    },

    "cloud_medium": {
        "input_per_1k_tokens": 0.00090,
        "output_per_1k_tokens": 0.00090,
        "description": "Cloud medium (Mixtral-8x7B / LLaMA-3-70B via API)",
    },

    "gpt4o": {
        "input_per_1k_tokens": 0.00500,
        "output_per_1k_tokens": 0.01500,
        "description": "OpenAI GPT-4o",
    },

    "claude_sonnet": {
        "input_per_1k_tokens": 0.00300,
        "output_per_1k_tokens": 0.01500,
        "description": "Anthropic Claude-3.5-Sonnet",
    },
}


def estimate_cost_per_1k(
    token_records: Sequence[Dict[str, int]],
    model_tier: str = "ollama_local",
) -> Dict[str, float]:
    """Estimate the dollar cost of running the pipeline at 1 000 queries."""
    if model_tier not in PRICING_TABLE:
        raise ValueError(
            f"Unknown model_tier '{model_tier}'. "
            f"Choose from: {list(PRICING_TABLE.keys())}"
        )

    if not token_records:
        pricing = PRICING_TABLE[model_tier]
        return {
            "model_tier": model_tier,
            "description": pricing["description"],
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "cost_per_query_usd": 0.0,
            "cost_per_1k_usd": 0.0,
            "cost_per_10k_usd": 0.0,
            "cost_per_100k_usd": 0.0,
        }

    pricing = PRICING_TABLE[model_tier]
    n = len(token_records)

    total_input = sum(r.get("input_tokens", 0) for r in token_records)
    total_output = sum(r.get("output_tokens", 0) for r in token_records)

    avg_input = total_input / n
    avg_output = total_output / n
    avg_total = avg_input + avg_output


    cost_per_query = (
        avg_input / 1000 * pricing["input_per_1k_tokens"]
        + avg_output / 1000 * pricing["output_per_1k_tokens"]
    )

    return {
        "model_tier": model_tier,
        "description": pricing["description"],
        "avg_input_tokens": round(avg_input, 1),
        "avg_output_tokens": round(avg_output, 1),
        "avg_total_tokens": round(avg_total, 1),
        "cost_per_query_usd": round(cost_per_query, 6),
        "cost_per_1k_usd": round(cost_per_query * 1_000, 4),
        "cost_per_10k_usd": round(cost_per_query * 10_000, 3),
        "cost_per_100k_usd": round(cost_per_query * 100_000, 2),
    }


def estimate_cost_all_tiers(
    token_records: Sequence[Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Run :func:`estimate_cost_per_1k` across all pricing tiers and return a combined dictionary keyed by tier name."""
    return {
        tier: estimate_cost_per_1k(token_records, model_tier=tier)
        for tier in PRICING_TABLE
    }


def compute_token_efficiency(
    answer: str,
    input_tokens: int,
    output_tokens: Optional[int] = None,
) -> Dict[str, float]:
    """Measure how efficiently the LLM converts input tokens into a useful answer."""
    if output_tokens is None:
        output_tokens = count_tokens_approx(answer)

    input_tokens = max(1, int(input_tokens))
    output_tokens = max(0, int(output_tokens))
    total = input_tokens + output_tokens

    output_ratio = round(output_tokens / input_tokens, 4)


    clamped_output = min(output_tokens, 512)
    efficiency_score = round(clamped_output / total, 4) if total > 0 else 0.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "output_ratio": output_ratio,
        "efficiency_score": efficiency_score,
    }


def aggregate_token_efficiency(
    records: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Average per-query token efficiency stats across a full evaluation run."""
    if not records:
        return {
            "sample_count": 0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "avg_output_ratio": 0.0,
            "avg_efficiency_score": 0.0,
        }

    n = len(records)
    keys = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "output_ratio",
        "efficiency_score",
    ]
    sums = {k: sum(float(r.get(k, 0)) for r in records) for k in keys}

    return {
        "sample_count": n,
        "avg_input_tokens": round(sums["input_tokens"] / n, 1),
        "avg_output_tokens": round(sums["output_tokens"] / n, 1),
        "avg_total_tokens": round(sums["total_tokens"] / n, 1),
        "avg_output_ratio": round(sums["output_ratio"] / n, 4),
        "avg_efficiency_score": round(sums["efficiency_score"] / n, 4),
    }


def build_token_record(
    prompt: str,
    context: str,
    answer: str,
) -> Dict[str, int]:
    """Build a token-count dict from raw text components."""
    input_tokens = count_tokens_approx(prompt) + count_tokens_approx(context)
    output_tokens = count_tokens_approx(answer)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


if __name__ == "__main__":

    sample_latencies = [1.2, 3.4, 2.1, 5.0, 1.8, 4.2, 2.9, 3.1, 1.5, 22.0]
    stats = compute_latency_stats(sample_latencies)
    print("Latency stats:")
    for k, v in stats.items():
        print(f"  {k:<10} {v}")


    sample_tokens = [
        {"input_tokens": 1800, "output_tokens": 350},
        {"input_tokens": 2100, "output_tokens": 420},
        {"input_tokens": 1500, "output_tokens": 280},
    ]
    print("\nCost estimates:")
    for tier, cost in estimate_cost_all_tiers(sample_tokens).items():
        print(
            f"  {cost['description']:<50} ${cost['cost_per_1k_usd']:.4f} / 1k queries"
        )


    answer_text = "The Supreme Court in Kesavananda Bharati held that Parliament cannot amend the basic structure of the Constitution."
    eff = compute_token_efficiency(answer=answer_text, input_tokens=1800)
    print("\nToken efficiency:")
    for k, v in eff.items():
        print(f"  {k:<22} {v}")
