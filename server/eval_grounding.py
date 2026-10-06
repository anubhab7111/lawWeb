"""Grounding NLI: collect real claim/evidence pairs, then gate an NLI model on them.

    python eval_grounding.py collect          # runs the chatbot; resumable, one line per question
    python eval_grounding.py premise          # how the slow LLM call splits into adjudication vs rewrite
"""

import argparse
import asyncio
import json
import lzma
import time
from pathlib import Path

from app.tools.grounding_verifier import CONTRADICTED, SUPPORTED, UNGROUNDED, _MAX_LLM_CORRECTIONS

RAW = Path.home() / ".cache" / "lawweb" / "grounding_collect.jsonl"
ILSIC_DEV = Path("/run/media/ushtro/anubhab_x9/lawweb/ilsic/Layman-new-dataset/FT-Layman-dev.jsonl.xz")
ILSIC_QUESTIONS = 10


def questions() -> list:
    from app.metrics.ground_truth import GROUND_TRUTH
    from app.metrics.ground_truth_extended import EXTENDED_GROUND_TRUTH

    out = [{"qid": f"gt{i}", "source": "ground_truth", "question": e["query"] if isinstance(e, dict) else e.query} for i, e in enumerate(GROUND_TRUTH)]
    out += [{"qid": f"gtx{i}", "source": "ground_truth_extended", "question": e["query"]}
            for i, e in enumerate(EXTENDED_GROUND_TRUTH)]
    picked = 0
    for i, line in enumerate(lzma.open(ILSIC_DEV, "rt")):
        text = " ".join(json.loads(line)["instruction"].split())
        if 20 <= len(text.split()) <= 80:
            out.append({"qid": f"ilsic_dev{i}", "source": "ilsic_dev", "question": text})
            picked += 1
            if picked == ILSIC_QUESTIONS:
                break
    return out


def in_to_fix(s) -> bool:
    """Mirror of ground_and_correct's selection, on the deterministic status."""
    return s.is_claim and (
        (s.det_status != SUPPORTED and (s.det_status in (CONTRADICTED, UNGROUNDED) or s.is_high_risk))
        or s.needs_llm
    )


def collect() -> None:
    import app.tools.grounding_verifier as gv
    from app.chatbot import get_chatbot

    calls = []
    original = gv.ground_and_correct

    async def recording(answer, rag, retrieved_sections=None, retrieved_context_text="", llm_invoke=None):
        llm_seconds = []

        async def timed(prompt):
            t0 = time.monotonic()
            try:
                return await llm_invoke(prompt)
            finally:
                llm_seconds.append(time.monotonic() - t0)

        text, report = await original(answer, rag, retrieved_sections, retrieved_context_text,
                                      timed if llm_invoke else None)
        fix = sorted((s for s in report.sentences if in_to_fix(s)), key=lambda s: s.overlap)[:_MAX_LLM_CORRECTIONS]
        fix_ids = {id(s) for s in fix}
        calls.append({
            "llm_seconds": round(sum(llm_seconds), 2),
            "llm_succeeded": report.llm_succeeded,
            "claims": [
                {
                    "text": s.text.strip(), "citations": s.citations, "high_risk": s.is_high_risk,
                    "det_status": s.det_status, "final_status": s.status, "overlap": round(s.overlap, 3),
                    "needs_llm": s.needs_llm, "sent_to_llm": id(s) in fix_ids, "outcome": s.outcome,
                    "evidence": s.evidence, "candidates": s.candidates,
                }
                for s in report.sentences if s.is_claim
            ],
        })
        return text, report

    gv.ground_and_correct = recording
    RAW.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in RAW.read_text().splitlines()} if RAW.exists() else set()
    bot = get_chatbot()
    todo = [q for q in questions() if q["qid"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    async def run_all():
        for q in todo:
            calls.clear()
            t0 = time.time()
            try:
                result = await bot.chat(message=q["question"], session_id=f"grounding-collect-{q['qid']}")
                row = {**q, "elapsed": round(time.time() - t0, 1), "intent": result.get("intent"),
                       "grounding_trace": (result.get("trace") or {}).get("grounding"), "verify_calls": list(calls)}
            except Exception as e:
                row = {**q, "elapsed": round(time.time() - t0, 1), "error": repr(e), "verify_calls": list(calls)}
            with RAW.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n = sum(len(c["claims"]) for c in calls)
            print(f"{q['qid']}: {row['elapsed']}s intent={row.get('intent')} claims={n}", flush=True)

    asyncio.run(run_all())


def premise() -> None:
    """Per verify call: does the slow LLM call carry rewrites (which NLI cannot remove)?"""
    rows = [json.loads(l) for l in RAW.read_text().splitlines()]
    calls = [c for r in rows for c in r.get("verify_calls", [])]
    llm_calls = [c for c in calls if c["llm_seconds"] > 0]
    adj_only, with_rewrite, adj_secs, rw_secs = 0, 0, 0.0, 0.0
    n_adj = n_rw = 0
    for c in llm_calls:
        sent = [x for x in c["claims"] if x["sent_to_llm"]]
        rw = [x for x in sent if x["det_status"] in (CONTRADICTED, UNGROUNDED)]
        n_rw += len(rw)
        n_adj += len(sent) - len(rw)
        if rw:
            with_rewrite += 1
            rw_secs += c["llm_seconds"]
        else:
            adj_only += 1
            adj_secs += c["llm_seconds"]
    print(f"questions {len(rows)}, verify calls {len(calls)}, with an LLM call {len(llm_calls)}")
    print(f"claims sent to LLM: {n_adj} adjudication-only, {n_rw} rewrite candidates (det CONTRADICTED/UNGROUNDED)")
    print(f"LLM calls with adjudication only (NLI removes the call): {adj_only}, {adj_secs:.0f}s total")
    print(f"LLM calls carrying a rewrite (call stays, shorter):     {with_rewrite}, {rw_secs:.0f}s total")
    all_claims = [x for c in calls for x in c["claims"]]
    print(f"all claims {len(all_claims)}; final status mix:",
          {s: sum(x["final_status"] == s for x in all_claims) for s in {x["final_status"] for x in all_claims}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "premise"])
    a = ap.parse_args()
    {"collect": collect, "premise": premise}[a.cmd]()
