import argparse
import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chatbot import get_chatbot

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class _Tee:
    """Mirror a stream to a log file so each run's full console output is saved."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()


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

# Extended prompts covering the domains added by the unified all-domain RAG
# index. Queries match app/metrics/ground_truth_extended.py exactly.
# Enabled with --extended.
EXTENDED_PROMPTS = [
    "What are the grounds for divorce under the Hindu Marriage Act?",
    "Can elderly parents claim maintenance from their children in India?",
    "How much maternity leave is a woman employee entitled to in India?",
    "What compensation must an employer pay when retrenching a workman?",
    "How can a financial creditor initiate corporate insolvency proceedings against a defaulting company?",
    "What is the requirement for independent directors on the board of a listed company?",
    "What are the conditions for claiming input tax credit under GST?",
    "Who is required to file an income tax return in India and what happens on late filing?",
    "What powers does the Central Government have under the Environment Protection Act to control pollution?",
    "How do I file a consumer complaint for a defective product and what relief can I get?",
    "What is the punishment for identity theft and cheating by personation online?",
    "What exclusive rights does a copyright owner have and how long does copyright protection last?",
    "What constitutes a corrupt practice in Indian elections?",
    "Can a homebuyer get a refund with interest if the builder delays possession under RERA?",
    "On what grounds can an arbitral award be set aside by a court?",
    "What are the functions and powers of the National Human Rights Commission?",
]


# ============================================================================
# Pass 1 — run the chatbot and collect raw answers
# ============================================================================


async def run_evaluation(prompts=None):
    """Run all test prompts through the chatbot and collect raw results."""
    chatbot = get_chatbot()
    results = []

    prompts = prompts or TEST_PROMPTS
    total = len(prompts)
    print(f"{'=' * 60}")
    print(f"  Legal Chatbot Evaluation — {total} prompts")
    print(f"{'=' * 60}\n")

    for i, prompt in enumerate(prompts, 1):
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


def save_csv(results: list, path: "str | Path"):
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
        True  -> uses the OpenRouter LLM-as-judge (slower, higher quality).
        False -> uses keyword heuristics only     (fast,  offline mode).
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
    metrics_csv_path = RESULTS_DIR / f"metrics_{timestamp}.csv"
    evaluator.save_csv(eval_results, metrics_csv_path)

    # Save full JSON (includes per-query reasoning strings + aggregate)
    metrics_json_path = RESULTS_DIR / f"metrics_{timestamp}.json"
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
  python tests/test_chatbot.py                        # basic run (no metrics)
  python tests/test_chatbot.py --metrics              # full 9-metric eval with LLM judge
  python tests/test_chatbot.py --metrics --no-llm-judge  # keyword heuristics only
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
        "--extended",
        action="store_true",
        default=False,
        help=(
            "Also run the extended new-domain prompts (family, labour, "
            "corporate, tax, environment, consumer, cyber/IP, election, "
            "property, commercial, human rights)."
        ),
    )
    parser.add_argument(
        "--extended-only",
        action="store_true",
        default=False,
        help="Run ONLY the extended new-domain prompts.",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=False,
        dest="no_llm_judge",
        help=(
            "Disable the OpenRouter LLM-as-judge and use keyword heuristics "
            "instead. Much faster and uses zero API quota; useful for "
            "offline / CI runs. Only applies when --metrics is also passed."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Evaluate only the first N prompts instead of the full set. "
            "Each prompt costs up to 4 LLM-judge calls, so use this to stay "
            "inside OpenRouter's free-tier daily budget (see "
            "OPENROUTER_DAILY_LIMIT in server/.env, default 50/day)."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the complete console output of this run to its own log file.
    run_log_path = RESULTS_DIR / f"run_{timestamp}.log"
    log_fh = open(run_log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"[log] Saving full run log to {run_log_path}\n")

    try:
        await _run(args, timestamp)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()
        print(f"[log] Full run log written to {run_log_path}")


async def _run(args: argparse.Namespace, timestamp: str) -> None:
    # ------------------------------------------------------------------
    # Pass 1: run the chatbot and collect raw answers + latencies
    # ------------------------------------------------------------------
    if args.extended_only:
        prompts = EXTENDED_PROMPTS
    elif args.extended:
        prompts = TEST_PROMPTS + EXTENDED_PROMPTS
    else:
        prompts = TEST_PROMPTS
    if args.sample is not None:
        prompts = prompts[: max(0, args.sample)]

    use_llm_judge = args.metrics and not args.no_llm_judge
    if use_llm_judge:
        from app.config import get_settings

        settings = get_settings()
        daily_limit = getattr(settings, "openrouter_daily_limit", 50)
        estimated_calls = len(prompts)  # one batched judge call per query
        print(
            f"[Budget] This run may make up to {estimated_calls} OpenRouter judge "
            f"calls against a daily budget of {daily_limit} "
            "(cached/repeated (query, context, answer) triples are free). "
            "Use --sample N to shrink the prompt set, or --no-llm-judge to "
            "skip the judge entirely.\n"
        )

    chatbot_results = await run_evaluation(prompts)

    # Save the basic CSV (same format as before, always written)
    basic_csv_path = RESULTS_DIR / f"eval_results_{timestamp}.csv"
    save_csv(chatbot_results, basic_csv_path)
    print_summary(chatbot_results)

    # ------------------------------------------------------------------
    # Pass 2 (optional): full metrics evaluation
    # ------------------------------------------------------------------
    if args.metrics:
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
