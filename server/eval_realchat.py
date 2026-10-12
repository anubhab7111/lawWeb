"""Run the 27 end-to-end chat cases (tests/gold/realchat_cases.json) through the chatbot; resumable per case.

    python eval_realchat.py run <label>        # uses the current settings (LLM_MODEL, FAST_LLM_MODEL, LLM_THINKING)
    python eval_realchat.py report <label> [<label> ...]

Results go to results/realchat_<label>.jsonl. Unload other Ollama models first so the timing is not shared.
"""

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, ".")

CASES = Path("tests/gold/realchat_cases.json")
OUT = Path("results")


def _path(label: str) -> Path:
    return OUT / f"realchat_{label}.jsonl"


def _rows(label: str) -> list:
    p = _path(label)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


async def run(label: str) -> None:
    from app.chatbot import get_chatbot
    from app.config import get_settings

    settings = get_settings()
    print(f"model={settings.llm_model} fast={settings.fast_llm_model} thinking={settings.llm_thinking}", flush=True)
    done = {r["id"] for r in _rows(label)}
    bot = get_chatbot()
    OUT.mkdir(exist_ok=True)
    for c in json.loads(CASES.read_text()):
        if c["id"] in done:
            continue
        sid = f"realchat-{label}-{c['id']}-{int(time.time())}"
        turns = []
        for k, msg in enumerate(c["turns"]):
            t0 = time.time()
            try:
                r = await bot.chat(message=msg, session_id=sid, document_content=c.get("doc") if k == 0 else None)
                err = None
            except Exception as e:  # noqa: BLE001
                r, err = {}, repr(e)
            dv = r.get("document_validation") or {}
            turns.append(
                {
                    "message": msg,
                    "seconds": round(time.time() - t0, 1),
                    "error": err,
                    "intent": r.get("intent"),
                    "crime_type": (r.get("crime_report") or {}).get("crime_type"),
                    "doc_type": dv.get("classified_type") if isinstance(dv, dict) else None,
                    "response": r.get("response") or "",
                }
            )
            print(f"{c['id']}[{k}] {turns[-1]['seconds']}s intent={turns[-1]['intent']} crime={turns[-1]['crime_type']} err={err}", flush=True)
        with _path(label).open("a") as f:
            f.write(json.dumps({"id": c["id"], "turns": turns}, ensure_ascii=False) + "\n")
    print("ALL DONE", flush=True)


def report(labels: list) -> None:
    cases = {c["id"]: c for c in json.loads(CASES.read_text())}
    print(f"{'label':22} {'cases':>5} {'intent':>7} {'crime':>6} {'errors':>6} {'no-match':>8} {'incomplete':>10} {'total s':>8}  median s by intent")
    for label in labels:
        rows = _rows(label)
        intent_ok = crime_ok = crime_n = errors = nomatch = incomplete = 0
        total, by_intent = 0.0, {}
        for r in rows:
            c, last = cases[r["id"]], r["turns"][-1]
            intent_ok += last["intent"] in c["intent"]
            if c.get("crime"):
                crime_n += 1
                crime_ok += last["crime_type"] in c["crime"]
            errors += any(t["error"] for t in r["turns"])
            nomatch += "No matching section" in last["response"] and bool(c.get("crime"))
            incomplete += "wasn't able to finish" in last["response"] or "cut short" in last["response"]
            seconds = sum(t["seconds"] for t in r["turns"])
            total += seconds
            by_intent.setdefault(c["intent"][0], []).append(seconds)
        med = {k: round(statistics.median(v), 1) for k, v in sorted(by_intent.items())}
        print(f"{label:22} {len(rows):5} {intent_ok:4}/{len(rows):<2} {crime_ok:3}/{crime_n:<2} {errors:6} {nomatch:8} {incomplete:10} {total:8.0f}  {med}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        asyncio.run(run(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        sys.exit(__doc__)
