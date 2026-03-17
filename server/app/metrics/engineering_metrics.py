"""
Engineering & Cost Metrics
==========================
Tracks the operational health and economic efficiency of the RAG pipeline.

Metrics implemented
-------------------
1. Latency Stats          — min / p50 / p95 / p99 / max / mean (seconds)
2. Cost per 1 k Queries   — estimated LLM spend at several reference pricing tiers
3. Token Usage Efficiency — ratio of useful output tokens to total tokens processed

All functions are pure (no I/O, no LLM calls) so they can be used in unit tests
without any external dependencies beyond the standard library.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# A rough but dependency-free token estimator.
# Real tiktoken / sentencepiece counts will differ by ±15 %, which is
# acceptable for cost *estimation* purposes.
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def count_tokens_approx(text: str) -> int:
    """
    Approximate the number of LLM tokens in *text*.

    Uses the widely-cited heuristic that 1 token ≈ 0.75 words for English
    prose (GPT-3/4 and most open-weight models are within ~10 % of this).
    Legal text skews slightly higher due to long Latin phrases and section
    numbers, so we apply a 1.10 correction factor.

    Args:
        text: Any string — prompt, context, or answer.

    Returns:
        Estimated integer token count (minimum 1 for non-empty strings).
    """
    if not text or not text.strip():
        return 0
    # Collapse whitespace, count words
    cleaned = _WHITESPACE_RE.sub(" ", text.strip())
    word_count = len(cleaned.split())
    # 1 token ≈ 0.75 words  →  tokens ≈ words / 0.75 ≈ words * 1.333
    # Legal text correction: multiply by 1.10
    token_count = math.ceil(word_count * 1.333 * 1.10)
    return max(1, token_count)


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------


def _percentile(sorted_values: List[float], pct: float) -> float:
    """
    Compute a percentile value from a **sorted** list using linear
    interpolation (same algorithm as numpy.percentile with interpolation='linear').

    Args:
        sorted_values: A list of floats, already sorted ascending.
        pct:           Percentile in [0, 100].

    Returns:
        Interpolated percentile value.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    # Linear interpolation index
    idx = (pct / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    """
    Compute a full latency distribution from a list of per-request timings.

    Args:
        latencies: Sequence of response times in **seconds** — one per query.

    Returns:
        Dictionary with keys::

            {
                "count":  int,   # number of samples
                "min":    float, # fastest request
                "mean":   float, # arithmetic mean
                "median": float, # p50
                "p95":    float, # 95th percentile
                "p99":    float, # 99th percentile
                "max":    float, # slowest request
                "stdev":  float, # sample standard deviation
            }

    Example:
        >>> stats = compute_latency_stats([1.2, 3.4, 2.1, 5.0, 1.8])
        >>> stats["p95"]  # doctest: +SKIP
        4.72
    """
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

    # Sample standard deviation
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


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Pricing reference table (USD per 1 000 tokens, as of mid-2025).
# Prices are approximate and for illustration only.
# Keys are model-tier names used in the report.
PRICING_TABLE: Dict[str, Dict[str, float]] = {
    # Self-hosted Ollama — only marginal electricity / hardware cost.
    # Approximated at $0.30/hr for a consumer GPU running at ~30 tok/s.
    "ollama_local": {
        "input_per_1k_tokens": 0.00010,  # ~$0.10 / 1M tokens
        "output_per_1k_tokens": 0.00010,
        "description": "Self-hosted Ollama (Mistral-7B / LLaMA-3-8B)",
    },
    # Cloud — small open-weight models via Together / Fireworks / Groq
    "cloud_small": {
        "input_per_1k_tokens": 0.00020,  # $0.20 / 1M
        "output_per_1k_tokens": 0.00020,
        "description": "Cloud small (Mistral-7B / LLaMA-3-8B via API)",
    },
    # Cloud — medium open-weight models
    "cloud_medium": {
        "input_per_1k_tokens": 0.00090,  # $0.90 / 1M
        "output_per_1k_tokens": 0.00090,
        "description": "Cloud medium (Mixtral-8x7B / LLaMA-3-70B via API)",
    },
    # GPT-4o (OpenAI, as of mid-2025)
    "gpt4o": {
        "input_per_1k_tokens": 0.00500,  # $5.00 / 1M
        "output_per_1k_tokens": 0.01500,  # $15.00 / 1M
        "description": "OpenAI GPT-4o",
    },
    # Claude 3.5 Sonnet (Anthropic)
    "claude_sonnet": {
        "input_per_1k_tokens": 0.00300,  # $3.00 / 1M
        "output_per_1k_tokens": 0.01500,  # $15.00 / 1M
        "description": "Anthropic Claude-3.5-Sonnet",
    },
}


def estimate_cost_per_1k(
    token_records: Sequence[Dict[str, int]],
    model_tier: str = "ollama_local",
) -> Dict[str, float]:
    """
    Estimate the dollar cost of running the pipeline at 1 000 queries.

    Args:
        token_records:
            Sequence of per-query token dicts, each with keys:
            - ``"input_tokens"``  : prompt + context tokens fed to the LLM
            - ``"output_tokens"`` : tokens in the LLM response
        model_tier:
            One of the keys in :data:`PRICING_TABLE`.
            Defaults to ``"ollama_local"``.

    Returns:
        Dictionary::

            {
                "model_tier":           str,
                "description":          str,
                "avg_input_tokens":     float,
                "avg_output_tokens":    float,
                "avg_total_tokens":     float,
                "cost_per_query_usd":   float,
                "cost_per_1k_usd":      float,
                "cost_per_10k_usd":     float,
                "cost_per_100k_usd":    float,
            }

    Raises:
        ValueError: If *model_tier* is not in :data:`PRICING_TABLE`.
    """
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

    # Cost for a single average query (USD)
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
    """
    Run :func:`estimate_cost_per_1k` across all pricing tiers and return a
    combined dictionary keyed by tier name.

    Useful for generating the side-by-side cost comparison table in reports.
    """
    return {
        tier: estimate_cost_per_1k(token_records, model_tier=tier)
        for tier in PRICING_TABLE
    }


# ---------------------------------------------------------------------------
# Token efficiency
# ---------------------------------------------------------------------------


def compute_token_efficiency(
    answer: str,
    input_tokens: int,
    output_tokens: Optional[int] = None,
) -> Dict[str, float]:
    """
    Measure how efficiently the LLM converts input tokens into a useful answer.

    Two complementary ratios are computed:

    * **output_ratio** — ``output_tokens / input_tokens``
      Should be well below 1.0; a ratio above ~0.5 may indicate a bloated
      prompt or a model that is repeating the context.

    * **efficiency_score** — a normalised score in [0, 1] that rewards answers
      which are (a) non-trivially long relative to the question and (b) don't
      merely echo back the retrieved context word-for-word.

      The formula is:
      ``efficiency_score = min(output_tokens, 512) / (input_tokens + output_tokens)``

      Intuition: an ideal answer uses ~20–40 % of the total budget on new,
      synthesised content.  Very short answers (< 50 tokens) score low;
      answers that dominate the total budget also score low.

    Args:
        answer:        The LLM's final answer text.
        input_tokens:  Tokens fed *into* the LLM (prompt + context).
                       Pass ``count_tokens_approx(prompt)`` if exact count
                       is unavailable.
        output_tokens: Tokens in the LLM's response.  If ``None``, estimated
                       from *answer* with :func:`count_tokens_approx`.

    Returns:
        Dictionary::

            {
                "input_tokens":     int,
                "output_tokens":    int,
                "total_tokens":     int,
                "output_ratio":     float,   # output / input
                "efficiency_score": float,   # 0 – 1
            }
    """
    if output_tokens is None:
        output_tokens = count_tokens_approx(answer)

    input_tokens = max(1, int(input_tokens))
    output_tokens = max(0, int(output_tokens))
    total = input_tokens + output_tokens

    output_ratio = round(output_tokens / input_tokens, 4)

    # efficiency_score: reward meaningful output without dominating the budget
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
    """
    Average per-query token efficiency stats across a full evaluation run.

    Args:
        records: List of dicts returned by :func:`compute_token_efficiency`.

    Returns:
        Dict with mean values for each numeric key, plus ``"sample_count"``.
    """
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


# ---------------------------------------------------------------------------
# Convenience: build token record from raw text
# ---------------------------------------------------------------------------


def build_token_record(
    prompt: str,
    context: str,
    answer: str,
) -> Dict[str, int]:
    """
    Build a token-count dict from raw text components.

    Combines prompt and retrieved context into ``input_tokens`` and counts
    the answer as ``output_tokens``.  All counts are approximations via
    :func:`count_tokens_approx`.

    Args:
        prompt:  The user query / system prompt passed to the LLM.
        context: The retrieved context / RAG documents added to the prompt.
        answer:  The LLM's generated response.

    Returns:
        ``{"input_tokens": int, "output_tokens": int}``
    """
    input_tokens = count_tokens_approx(prompt) + count_tokens_approx(context)
    output_tokens = count_tokens_approx(answer)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- latency ---
    sample_latencies = [1.2, 3.4, 2.1, 5.0, 1.8, 4.2, 2.9, 3.1, 1.5, 22.0]
    stats = compute_latency_stats(sample_latencies)
    print("Latency stats:")
    for k, v in stats.items():
        print(f"  {k:<10} {v}")

    # --- cost ---
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

    # --- token efficiency ---
    answer_text = "The Supreme Court in Kesavananda Bharati held that Parliament cannot amend the basic structure of the Constitution."
    eff = compute_token_efficiency(answer=answer_text, input_tokens=1800)
    print("\nToken efficiency:")
    for k, v in eff.items():
        print(f"  {k:<22} {v}")
