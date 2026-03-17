"""
Chatbot evaluation script.
Runs hardcoded legal prompts through the chatbot and saves results to CSV.
Optionally runs the full 9-metric evaluation suite via --metrics flag.

Usage:
    cd server/
    source myenv/bin/activate

    # Basic run (answers + latency only, same as before):
    python test_chatbot.py

    # Full metrics evaluation with LLM judge:
    python test_chatbot.py --metrics

    # Full metrics evaluation using keyword heuristics only (no Ollama judge):
    python test_chatbot.py --metrics --no-llm-judge
"""

import argparse
import asyncio
import csv
import time
from datetime import datetime

from app.chatbot import get_chatbot

from dotenv import load_dotenv

load_dotenv()
# ============================================================================
# Test prompts — covering the domains that previously had accuracy issues
# ============================================================================
TEST_PROMPTS = [
    "Can Parliament pass a law restricting social media speech citing \u201cpublic order\u201d? How would courts test its constitutionality under Article 19?",
    "Is the Right to Privacy absolute in India? Under what circumstances can the State legally conduct surveillance?",
    "Can a State government refuse to implement a Central law? What remedies exist?",
    "How does the \u201cbasic structure doctrine\u201d limit constitutional amendments?",
    "Can an FIR be quashed by the High Court? On what grounds?",
    "Is anticipatory bail available for economic offences?",
    "Can a criminal case proceed if the complainant withdraws?",
    "Does marital rape constitute an offence in India? Explain the legal position.",
    "Can cryptocurrency transactions attract criminal liability under existing Indian laws?",
    "If someone's private photos are shared online without consent, what legal remedies are available?",
    "Who is liable if an AI system causes financial loss — developer, deployer, or user?",
    "Are WhatsApp chats admissible as evidence in Indian courts?",
    "Is a contract enforceable if signed under economic pressure but without explicit coercion?",
    "Can an oral agreement be legally binding in India?",
    "What happens if one party breaches a contract but claims \u201cforce majeure\u201d?",
    "Is a non-compete clause valid after employment ends?",
    "Can ancestral property be sold without consent of all legal heirs?",
    "What legal rights does a live-in partner have over shared property?",
]


# ============================================================================
# Pass 1 — run the chatbot and collect raw answers
# ============================================================================


async def run_evaluation():
    """Run all test prompts through the chatbot and collect raw results."""
    chatbot = get_chatbot()
    results = []

    total = len(TEST_PROMPTS)
    print(f"{'=' * 60}")
    print(f"  Legal Chatbot Evaluation — {total} prompts")
    print(f"{'=' * 60}\n")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{total}] {prompt[:80]}...")
        start = time.time()

        try:
            # Use a fresh session per prompt to avoid context leakage
            session_id = f"eval_{i}"
            result = await chatbot.chat(message=prompt, session_id=session_id)
            answer = result.get("response", "ERROR: No response")
            intent = result.get("intent", "unknown")
            elapsed = round(time.time() - start, 2)
            print(
                f"        intent={intent}  time={elapsed}s  len={len(answer)} chars\n"
            )
        except Exception as e:
            answer = f"ERROR: {e}"
            intent = "error"
            elapsed = round(time.time() - start, 2)
            print(f"        FAILED: {e}\n")

        results.append(
            {
                "query": prompt,
                "intent": intent,
                "answer": answer,
                "response_time_s": elapsed,
            }
        )

    return results


# ============================================================================
# Basic CSV save (Pass 1 only)
# ============================================================================


def save_csv(results: list, path: str):
    """Save raw chatbot results to a CSV file."""
    fieldnames = ["query", "intent", "answer", "response_time_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


# ============================================================================
# Basic summary (Pass 1 only)
# ============================================================================


def print_summary(results: list):
    """Print a quick summary of the chatbot run (no metrics)."""
    total = len(results)
    errors = sum(1 for r in results if r["answer"].startswith("ERROR"))
    avg_time = sum(r["response_time_s"] for r in results) / total if total else 0
    intents = {}
    for r in results:
        intents[r["intent"]] = intents.get(r["intent"], 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Total prompts : {total}")
    print(f"  Errors        : {errors}")
    print(f"  Avg time      : {avg_time:.2f}s")
    print(f"  Intents       : {intents}")
    print(f"{'=' * 60}\n")


# ============================================================================
# Pass 2 — full 9-metric evaluation
# ============================================================================


async def run_metrics_evaluation(
    chatbot_results: list,
    timestamp: str,
    use_llm_judge: bool = True,
) -> None:
    """
    Run the MetricsEvaluator over the chatbot results and save both
    a metrics CSV and a full JSON report.

    Parameters
    ----------
    chatbot_results : list[dict]
        Output of run_evaluation().
    timestamp : str
        Timestamp string used for output filenames.
    use_llm_judge : bool
        True  -> uses local Ollama LLM-as-judge (slower, higher quality).
        False -> uses keyword heuristics only   (fast,  offline mode).
    """
    try:
        from app.metrics.evaluator import MetricsEvaluator
    except ImportError as e:
        print(f"\n[Metrics] Import error: {e}")
        print("[Metrics] Skipping metrics evaluation.\n")
        return

    print("\n" + "=" * 60)
    print("  PASS 2 — Full 9-Metric Evaluation")
    print("=" * 60)

    evaluator = MetricsEvaluator(
        use_llm_judge=use_llm_judge,
        rag_k=5,
        max_concurrent_judge_calls=3,
    )

    # Run all metrics
    eval_results = await evaluator.run(chatbot_results)

    # Print the rich report to stdout
    evaluator.print_report(eval_results)

    # Save detailed CSV (one row per query, all 9 metrics as columns)
    metrics_csv_path = f"metrics_{timestamp}.csv"
    evaluator.save_csv(eval_results, metrics_csv_path)

    # Save full JSON (includes per-query reasoning strings + aggregate)
    metrics_json_path = f"metrics_{timestamp}.json"
    evaluator.save_json(eval_results, metrics_json_path)

    print(f"\n[Metrics] Reports written:")
    print(f"  CSV  -> {metrics_csv_path}")
    print(f"  JSON -> {metrics_json_path}\n")


# ============================================================================
# Entry point
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legal chatbot evaluation script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_chatbot.py                        # basic run (no metrics)
  python test_chatbot.py --metrics              # full 9-metric eval with LLM judge
  python test_chatbot.py --metrics --no-llm-judge  # keyword heuristics only
        """,
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        default=True,
        help=(
            "Run the full 9-metric evaluation suite (Hit Rate@k, MRR, "
            "Context Precision, Faithfulness, Answer Relevance, Context Recall, "
            "Latency, Cost, Token Efficiency)."
        ),
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=False,
        dest="no_llm_judge",
        help=(
            "Disable the Ollama LLM-as-judge and use keyword heuristics instead. "
            "Much faster; useful for offline / CI runs. "
            "Only applies when --metrics is also passed."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Pass 1: run the chatbot and collect raw answers + latencies
    # ------------------------------------------------------------------
    chatbot_results = await run_evaluation()

    # Save the basic CSV (same format as before, always written)
    basic_csv_path = f"eval_results_{timestamp}.csv"
    save_csv(chatbot_results, basic_csv_path)
    print_summary(chatbot_results)

    # ------------------------------------------------------------------
    # Pass 2 (optional): full metrics evaluation
    # ------------------------------------------------------------------
    if args.metrics:
        use_llm_judge = not args.no_llm_judge
        await run_metrics_evaluation(
            chatbot_results=chatbot_results,
            timestamp=timestamp,
            use_llm_judge=use_llm_judge,
        )
    else:
        print(
            "\nTip: re-run with --metrics to compute Hit Rate@k, MRR, "
            "Faithfulness, Answer Relevance, Context Recall, Latency stats, "
            "Cost estimates, and Token Efficiency.\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
