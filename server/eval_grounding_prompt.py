"""A/B the grounding-correction prompt on the real LLM verify batches recorded by
`eval_grounding.py collect`: seconds per call, and each prompt's verdicts scored against
the labelled claims in tests/gold/grounding.jsonl.

    python eval_grounding_prompt.py
"""

import asyncio
import json
import subprocess
import sys
import time
import types

from app.tools import grounding_verifier as gv
from eval_grounding import RAW, load_gold


def old_module():
    src = subprocess.check_output(["git", "show", "origin/main:server/app/tools/grounding_verifier.py"], text=True)
    mod = types.ModuleType("grounding_verifier_main")
    sys.modules[mod.__name__] = mod
    exec(compile(src, "grounding_verifier_main", "exec"), mod.__dict__)
    return mod


def batches():
    for line in RAW.read_text().splitlines():
        row = json.loads(line)
        for ci, call in enumerate(row.get("verify_calls", [])):
            sent = [(f"{row['qid']}.{ci}.{k}", c) for k, c in enumerate(call["claims"]) if c["sent_to_llm"]]
            if sent:
                yield sent


async def main():
    from app.chatbot import get_grounding_correction_llm, invoke_llm_safely, strip_reasoning_tags

    llm = get_grounding_correction_llm()

    async def invoke(prompt):
        return strip_reasoning_tags(await invoke_llm_safely(llm, prompt, stream=False))

    arms = (("old", old_module()), ("new", gv))
    seconds = {name: 0.0 for name, _ in arms}
    verdicts = {name: {} for name, _ in arms}
    for batch in batches():
        for name, mod in arms:
            sentences = [mod.SentenceGrounding(text=c["text"], start=0, end=len(c["text"]), evidence=c["evidence"],
                                               det_status=c["det_status"], is_claim=True) for _, c in batch]
            t0 = time.monotonic()
            try:
                out = await mod._llm_adjudicate_and_correct(sentences, invoke)
            except Exception as e:
                out = {}
                print(f"{name}: {e!r}")
            seconds[name] += time.monotonic() - t0
            for i, (status, _) in out.items():
                verdicts[name][batch[i][0]] = status

    gold = {r["cid"]: r for r in load_gold()}
    print("LLM seconds:", {k: round(v) for k, v in seconds.items()})
    for split in ("dev", "test", None):
        for name in verdicts:
            ids = [c for c in verdicts[name] if c in gold and (split is None or gold[c]["split"] == split)]
            right = sum((verdicts[name][c] == gv.SUPPORTED) == (gold[c]["label"] == gv.SUPPORTED) for c in ids)
            print(f"{split or 'all'} {name}: verdict accuracy {right}/{len(ids)}")


if __name__ == "__main__":
    asyncio.run(main())
