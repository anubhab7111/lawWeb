"""
Evaluate the legal chatbot against IL-TUR (Indian Legal Text Understanding &
Reasoning), a public Indian legal-NLP benchmark — specifically its `lsi`
(Legal Statute Identification) subtask.

Unlike test_chatbot.py's fixed, hand-curated prompt lists, this script draws
a fresh random sample of real case-fact patterns from IL-TUR every run (use
--seed for a reproducible sample), so the chatbot gets exercised against
cases it wasn't tuned against. Reuses test_chatbot.py's chatbot-invocation,
metrics-evaluation, and reporting machinery — only the prompt/ground-truth
source differs.

IL-TUR is gated on HuggingFace: accept the license at
https://huggingface.co/datasets/Exploration-Lab/IL-TUR and set
HUGGINGFACE_TOKEN in server/.env before running this script.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.metrics.ground_truth import GROUND_TRUTH
from app.metrics.iltur_loader import (
    iltur_case_to_ground_truth,
    iltur_case_to_prompt,
    sample_iltur_cases,
)

from dotenv import load_dotenv

load_dotenv()

from test_chatbot import (  # noqa: E402
    RESULTS_DIR,
    _Tee,
    print_summary,
    run_evaluation,
    run_metrics_evaluation,
    save_csv,
)


def prepare_iltur_prompts(sample_size: int, seed: "int | None"):
    """Sample IL-TUR cases, register their ground truth, and return prompts."""
    print(f"[IL-TUR] Loading '{iltur_case_to_prompt.__module__}' lsi test split...")
    rows = sample_iltur_cases(sample_size, seed=seed)

    prompts = []
    print(f"[IL-TUR] Sampled {len(rows)} cases (seed={seed})")
    for row in rows:
        prompt = iltur_case_to_prompt(row)
        gt_entry = iltur_case_to_ground_truth(row, prompt)
        GROUND_TRUTH.append(gt_entry)
        prompts.append(prompt)
        print(f"  id={row.get('id')} sections={gt_entry.get('relevant_sections')}")

    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the legal chatbot against a random IL-TUR (lsi) sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_chatbot_iltur.py                       # 20 fresh random cases, full metrics
  python tests/test_chatbot_iltur.py --sample-size 5 --no-llm-judge   # quick offline smoke test
  python tests/test_chatbot_iltur.py --seed 42              # reproducible sample for before/after comparisons
        """,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        metavar="N",
        help="Number of IL-TUR cases to sample (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Seed the sample for reproducibility. Default: fresh random sample every run.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        default=True,
        help="Run the full 9-metric evaluation suite (default: on).",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=False,
        dest="no_llm_judge",
        help=(
            "Disable the OpenRouter LLM-as-judge and use keyword heuristics "
            "instead. Much faster and uses zero API quota."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_log_path = RESULTS_DIR / f"iltur_run_{timestamp}.log"
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
    prompts = prepare_iltur_prompts(args.sample_size, args.seed)

    use_llm_judge = args.metrics and not args.no_llm_judge
    if use_llm_judge:
        from app.config import get_settings

        settings = get_settings()
        daily_limit = getattr(settings, "openrouter_daily_limit", 50)
        print(
            f"[Budget] This run may make up to {len(prompts)} OpenRouter judge "
            f"calls against a daily budget of {daily_limit}. Use --sample-size N "
            "to shrink the set, or --no-llm-judge to skip the judge entirely.\n"
        )

    chatbot_results = await run_evaluation(prompts)

    basic_csv_path = RESULTS_DIR / f"iltur_eval_results_{timestamp}.csv"
    save_csv(chatbot_results, basic_csv_path)
    print_summary(chatbot_results)

    if args.metrics:
        await run_metrics_evaluation(
            chatbot_results=chatbot_results,
            timestamp=f"iltur_{timestamp}",
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
