#!/usr/bin/env python3
"""
generate_case_firac.py — backfill the `firac` block on app/data/case_law/*.json.

Each case JSON already carries ontology-seeded doctrines/statutes_cited but no
per-case Facts/Issues/Rules/holding/ratio summary. This script calls the LLM
once per case (case_firac_extractor.extract_case_firac) and writes the result
into the same file under a new `firac` key, alongside every existing field.

Idempotent/resumable: a case that already has a truthy `firac` key is skipped
unless --force, so an interrupted or partially-failed run can just be
re-invoked. This is an offline batch job (unlike the interactive get_llm()
singleton) so it uses its own longer timeout/num_predict/keep_alive tuned for
summarizing an ~8000-char judgment excerpt rather than a chat turn.

Usage (conda env legal_chatbot_env, run from server/):
    python generate_case_firac.py            # backfill all cases missing firac
    python generate_case_firac.py --limit 3  # smoke test
    python generate_case_firac.py --force    # regenerate everything
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from langchain_ollama import ChatOllama

from app.config import get_settings
from app.tools.case_firac_extractor import extract_case_firac

CASE_LAW_DIR = _SERVER_DIR / "app" / "data" / "case_law"


def _build_llm() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=8192,  # head+tail judgment excerpt (~8000 chars) + prompt scaffolding
        num_predict=1536,  # multi-issue Constitution Bench cases need headroom
        timeout=180.0,  # offline batch job — latency doesn't matter, don't spuriously abort
        reasoning=False,
        keep_alive="15m",  # stay loaded across the whole batch
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    case_files = sorted(CASE_LAW_DIR.glob("*.json"))
    if not case_files:
        print(f"No case files found in {CASE_LAW_DIR}")
        sys.exit(1)

    llm = _build_llm()
    generated = 0
    skipped = 0
    failed = 0
    processed = 0

    for path in case_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("firac") and not args.force:
            skipped += 1
            continue
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        case_name = data.get("case_name", path.stem)
        print(f"[{processed}] Extracting FIRAC: {case_name}")
        t0 = time.monotonic()
        try:
            result = await extract_case_firac(
                case_name, data.get("citation", ""), data.get("text", ""), llm=llm
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = None

        elapsed = time.monotonic() - t0
        if result and (result.issues or result.holding):
            data["firac"] = result.to_json_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(
                f"  saved ({elapsed:.0f}s): {len(result.issues)} issues, "
                f"domain={result.domain!r}"
            )
            generated += 1
        else:
            print(f"  FAILED ({elapsed:.0f}s): empty issues/holding — left for retry")
            failed += 1

    print(
        f"\nDone. generated={generated} skipped(existing)={skipped} "
        f"failed={failed} total={len(case_files)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
