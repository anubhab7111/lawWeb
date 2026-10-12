"""Benchmark small Ollama models as the chatbot's LLM: fit on the GPU, answer quality and latency.

    python eval_models.py fit                      # which candidates load fully on the GPU
    python eval_models.py capture                  # record the prompts the pipeline sends (baseline model)
    python eval_models.py replay [--models a,b] [--screen] [--runs 2]
    python eval_models.py score                    # gates, quality, latency, recommendation
    python eval_models.py blind | unblind          # anonymised review sheet and its key

Method, metric weights and gates are in tests/gold/README.md and app/metrics/model_bench.py; results go to
results/model_bench/. Replay sends every model the identical captured prompts, so retrieval is held constant.
"""

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, ".")

from app.metrics import model_bench as mb  # noqa: E402
from eval_realchat import fir_text, load_cases  # noqa: E402

GOLD = Path("tests/gold/model_bench.jsonl")
OUT = Path("results/model_bench")
CALLS = Path("tests/gold/model_bench_calls.jsonl")
FIT = OUT / "fit.json"
REPLAY = OUT / "replay.jsonl"
SCORES = OUT / "scores.json"

BASELINE = "qwen3-4b-thinking"
# inline_think: the model's template always opens a reasoning block, so content is reasoning + "</think>" + answer
# even with think=false (which is what the app sends). think=True asks Ollama for a separate thinking field.
MODELS = {
    "qwen3-4b-thinking": {"tag": "qwen3:4b", "think": False, "inline_think": True},
    "qwen3-4b-instruct": {"tag": "qwen3:4b-instruct", "think": False},
    "qwen3.5-2b": {"tag": "qwen3.5:2b", "think": False},
    "qwen3.5-2b-think": {"tag": "qwen3.5:2b", "think": True},
    "qwen3.5-4b": {"tag": "qwen3.5:4b", "think": False},
    "qwen3.5-4b-think": {"tag": "qwen3.5:4b", "think": True},
    "granite4.2-3b": {"tag": "granite4.2:3b", "think": False},
    "phi4-mini": {"tag": "phi4-mini", "think": False},
    "llama3.2-3b": {"tag": "llama3.2:3b", "think": False},
    "qwen2.5-3b": {"tag": "qwen2.5:3b-instruct", "think": False},
    "nemotron-mini-4b": {"tag": "nemotron-mini:4b", "think": False},
    "ministral-3-3b": {"tag": "ministral-3:3b", "think": False},
    "gemma3-4b": {"tag": "gemma3:4b", "think": False},
}
NUM_CTX = 8192
SEED0 = 1000
SCREEN_PER_STRATUM = {"A": 4, "B": 2, "C": 2, "D": 2, "E": 1, "F": 2}
BLIND_PER_STRATUM = {"A": 6, "B": 3, "C": 2, "D": 3, "E": 1}
BLIND_MAX_CHARS = 3000


def ollama_url() -> str:
    from app.config import get_settings

    return get_settings().ollama_base_url


def load_items() -> list:
    return [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


_FIR_MARK_RE = re.compile(r"\{\{icdar_fir:(.+?)\}\}")


def load_calls() -> list:
    """Captured prompts, with the unlicensed FIR text (stored as a reference) filled in from the local cache."""
    calls = read_jsonl(CALLS)
    for c in calls:
        c["prompt"] = _FIR_MARK_RE.sub(lambda m: fir_text(m.group(1)), c["prompt"])
    return calls


def mask_fir(prompt: str) -> str:
    """Inverse of load_calls(): swap any referenced FIR text in a captured prompt back to its reference."""
    from app.prompts import sanitize_untrusted_document
    from eval_realchat import CASES as cases_path

    for c in json.loads(cases_path.read_text()):
        ref = c.get("doc_ref")
        if ref:
            text = sanitize_untrusted_document(fir_text(ref["id"]))
            prompt = prompt.replace(text, "{{icdar_fir:" + ref["id"] + "}}")
    return prompt


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- capture

_item_var: contextvars.ContextVar = contextvars.ContextVar("bench_item", default=None)
_captured: list = []


def _role(llm) -> str:
    if llm.format:
        return "tiebreak" if llm.num_predict <= 64 else "correction"
    return {128: "fast", 900: "prose"}.get(llm.num_predict, "main")


def _record(llm, messages) -> None:
    item = _item_var.get()
    if item is None:
        return
    _captured.append(
        {
            "item": item,
            "seq": sum(c["item"] == item for c in _captured),
            "role": _role(llm),
            "prompt": messages[0].content,
            "num_predict": llm.num_predict,
            "temperature": llm.temperature,
            "format": llm.format or None,
        }
    )


def _patch_chat_ollama() -> None:
    from langchain_ollama import ChatOllama

    orig_ainvoke, orig_astream = ChatOllama.ainvoke, ChatOllama.astream

    async def ainvoke(self, input, *a, **k):
        _record(self, input)
        return await orig_ainvoke(self, input, *a, **k)

    async def astream(self, input, *a, **k):
        _record(self, input)
        async for chunk in orig_astream(self, input, *a, **k):
            yield chunk

    ChatOllama.ainvoke, ChatOllama.astream = ainvoke, astream


async def _capture() -> None:
    from app import chatbot as cb

    _patch_chat_ollama()
    cases = {c["id"]: c for c in load_cases()}
    done = {c["item"] for c in read_jsonl(CALLS)}
    bot = None
    for item in load_items():
        if item["id"] in done:
            continue
        _item_var.set(item["id"])
        before = len(_captured)
        if item["stratum"] == "B":
            # No-retrieval prompt built by the app's own builder, so it does not depend on what retrieval misses today.
            prompt, _ = cb._build_answer_prompt(
                {"current_input": item["message"], "messages": [], "tool_results": {}, "rag_succeeded": False}
            )
            _captured.append(
                {"item": item["id"], "seq": 0, "role": "main", "prompt": prompt,
                 "num_predict": cb.LLM_NUM_PREDICT, "temperature": cb.get_llm().temperature, "format": None}
            )
        elif item["stratum"] == "F":
            await cb._resolve_ambiguity_with_llm(frozenset(item["contenders"]), item["message"])
        else:
            bot = bot or cb.get_chatbot()
            doc = cases[item["case"]]["doc"] if item.get("case") else None
            await bot.chat(message=item["message"], session_id=f"bench-{item['id']}-{int(time.time())}", document_content=doc)
        calls = _captured[before:]
        for c in calls:
            append_jsonl(CALLS, {**c, "prompt": mask_fir(c["prompt"])})
        print(f"{item['id']}: {[c['role'] for c in calls]}", flush=True)
    print("CAPTURE DONE", flush=True)


# ---------------------------------------------------------------- ollama

def unload_all(client: httpx.Client) -> None:
    for m in client.get("/api/ps").json().get("models", []):
        client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
    for _ in range(40):
        if not client.get("/api/ps").json().get("models"):
            return
        time.sleep(0.5)


def chat_once(client: httpx.Client, cfg: dict, call: dict, seed: int) -> dict:
    """One streamed call with production options; stops at the role's production timeout."""
    limit = mb.ROLE_TIMEOUTS[call["role"]]
    body = {
        "model": cfg["tag"],
        "messages": [{"role": "user", "content": call["prompt"]}],
        "stream": True,
        "think": cfg["think"],
        "keep_alive": "30m",
        "options": {"num_ctx": NUM_CTX, "num_predict": call["num_predict"], "temperature": call["temperature"], "seed": seed},
    }
    if call.get("format"):
        body["format"] = call["format"]
    content, thinking_chunks, visible_chunks = "", 0, 0
    first_visible, final, error, timed_out = None, {}, None, False
    # A schema-constrained call cannot emit a reasoning block, so its content is the answer even on a thinking build.
    inline = bool(cfg.get("inline_think")) and not call.get("format")
    closed = not inline
    t0 = time.monotonic()
    try:
        with client.stream("POST", "/api/chat", json=body, timeout=httpx.Timeout(limit, connect=10)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                if msg.get("thinking"):
                    thinking_chunks += 1
                piece = msg.get("content") or ""
                if piece:
                    content += piece
                    if closed:
                        visible_chunks += 1
                        if first_visible is None:
                            first_visible = time.monotonic() - t0
                    else:
                        thinking_chunks += 1
                        closed = "</think>" in content
                if chunk.get("done"):
                    final = chunk
                    break
                if time.monotonic() - t0 > limit:
                    timed_out = True
                    break
    except httpx.TimeoutException:
        timed_out = True
    except Exception as e:  # noqa: BLE001
        error = repr(e)
    total = time.monotonic() - t0

    if inline:
        # As in the app: no closing tag means the answer never started.
        visible = content.rsplit("</think>", 1)[-1].strip() if "</think>" in content else ""
    else:
        visible = content.rsplit("</think>", 1)[-1].strip()
    eval_count = final.get("eval_count") or (thinking_chunks + visible_chunks)
    eval_ns = final.get("eval_duration") or 0
    return {
        "visible": visible,
        "seconds": round(total, 2),
        "ttft_visible": round(first_visible, 2) if first_visible is not None else None,
        "load_seconds": round((final.get("load_duration") or 0) / 1e9, 2),
        "prompt_tokens": final.get("prompt_eval_count"),
        "output_tokens": eval_count,
        "thinking_tokens": thinking_chunks,
        "tok_per_s": round(eval_count / eval_ns * 1e9, 1) if eval_ns else None,
        "done_reason": final.get("done_reason") or ("timeout" if timed_out else "error"),
        "timed_out": timed_out,
        "error": error,
    }


def load_and_check(client: httpx.Client, key: str) -> dict:
    """Load one model alone at production context size and report whether it sits fully on the GPU."""
    cfg = MODELS[key]
    unload_all(client)
    probe = {"role": "main", "prompt": "Reply with the single word OK.", "num_predict": 400, "temperature": 0.1}
    t0 = time.monotonic()
    r = chat_once(client, cfg, probe, SEED0)
    loaded = client.get("/api/ps").json().get("models", [])
    m = loaded[0] if loaded else {}
    return {
        "model": key,
        "tag": cfg["tag"],
        "loaded": bool(loaded) and not r["error"],
        "size_gb": round(m.get("size", 0) / 1e9, 2),
        "vram_gb": round(m.get("size_vram", 0) / 1e9, 2),
        "fits": bool(loaded) and m.get("size_vram") == m.get("size"),
        "load_seconds": round(time.monotonic() - t0, 1),
        "error": r["error"],
    }


def cmd_fit(args) -> None:
    results = json.loads(FIT.read_text()) if FIT.exists() else {}
    speed = {"role": "main", "num_predict": 300, "temperature": 0.1,
             "prompt": "Explain in about 150 words what a First Information Report is under Indian criminal procedure."}
    with httpx.Client(base_url=ollama_url(), timeout=120) as client:
        for key in args.models or MODELS:
            row = load_and_check(client, key)
            if row["loaded"]:
                r = chat_once(client, MODELS[key], speed, SEED0)
                row.update(tok_per_s=r["tok_per_s"], probe_thinking_tokens=r["thinking_tokens"],
                           probe_leaks_reasoning="</think>" in r["visible"] or mb.leaked(r["visible"]),
                           probe_head=r["visible"][:120])
            results[key] = row
            print(json.dumps(row), flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            FIT.write_text(json.dumps(results, indent=1))
        unload_all(client)


# ---------------------------------------------------------------- replay

def screen_ids(items: list) -> set:
    picked, seen = set(), {}
    for it in items:
        n = seen.get(it["stratum"], 0)
        if n < SCREEN_PER_STRATUM.get(it["stratum"], 0):
            picked.add(it["id"])
            seen[it["stratum"]] = n + 1
    return picked


def cmd_replay(args) -> None:
    items = load_items()
    wanted = screen_ids(items) if args.screen else {it["id"] for it in items}
    calls = [c for c in load_calls() if c["item"] in wanted]
    fit = json.loads(FIT.read_text())
    done = {(r["model"], r["item"], r["seq"], r["run"]) for r in read_jsonl(REPLAY)}
    with httpx.Client(base_url=ollama_url(), timeout=120) as client:
        for key in args.models or [k for k in MODELS if fit.get(k, {}).get("fits")]:
            todo = [(c, run) for run in range(args.runs) for c in calls if (key, c["item"], c["seq"], run) not in done]
            if not todo:
                continue
            state = load_and_check(client, key)
            if not state["fits"]:
                print(f"{key}: not fully on the GPU ({state}) — skipped", flush=True)
                continue
            for i, (c, run) in enumerate(todo):
                r = chat_once(client, MODELS[key], c, SEED0 + run)
                append_jsonl(REPLAY, {"model": key, "item": c["item"], "seq": c["seq"], "role": c["role"], "run": run, **r})
                print(f"{key} {i + 1}/{len(todo)} {c['item']}#{c['seq']} {c['role']} run{run} {r['seconds']}s "
                      f"out={r['output_tokens']} think={r['thinking_tokens']} {r['done_reason']}", flush=True)
        unload_all(client)
    print("REPLAY DONE", flush=True)


# ---------------------------------------------------------------- score

def answer_seq(item: dict, calls: list) -> int:
    """The call whose output is the user-facing answer: the last main call for documents (ANALYZE), else the first."""
    mains = [c["seq"] for c in calls if c["role"] == "main"]
    return mains[-1] if item["stratum"] == "E" else mains[0]


def _pct(values: list, q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def _failed(r: dict, cfg: dict) -> bool:
    return bool(r["error"] or r["timed_out"] or not r["visible"] or (r["role"] == "main" and r["done_reason"] != "stop"))


async def _rag():
    from app.tools.unified_legal_rag import get_unified_rag_system

    rag = get_unified_rag_system()
    if not rag.initialized:
        await rag.initialize()
    return rag


def cmd_score(args) -> None:
    rag = asyncio.run(_rag())
    items = {it["id"]: it for it in load_items()}
    calls = {}
    for c in load_calls():
        calls.setdefault(c["item"], []).append(c)
    fit = json.loads(FIT.read_text())
    rows = read_jsonl(REPLAY)
    out = {}
    for key in MODELS:
        mine = [r for r in rows if r["model"] == key]
        if not mine:
            continue
        per_item, metric_sums, answers, json_valid, json_correct, leaks = {}, {}, [], [], [], 0
        for r in mine:
            item = items[r["item"]]
            if r["role"] in ("tiebreak", "correction"):
                ok = mb.json_call_ok(r["role"], r["visible"], item.get("gold", {}).get("intent"), item.get("contenders", ()))
                json_valid.append(ok["valid"])
                if r["role"] == "tiebreak":
                    json_correct.append(ok["correct"])
            if item["stratum"] == "F" or r["seq"] != answer_seq(item, calls[r["item"]]):
                continue
            prompt = next(c["prompt"] for c in calls[r["item"]] if c["seq"] == r["seq"])
            m = mb.answer_metrics(r["visible"], item, prompt, r["done_reason"], rag)
            q = mb.quality(m)
            leaks += mb.leaked(r["visible"])
            answers.append({**r, "metrics": m, "quality": q})
            per_item.setdefault(r["item"], []).append(q)
            for k, v in m.items():
                if v is not None:
                    metric_sums.setdefault((item["stratum"], k), []).append(v)
        item_quality = {i: statistics.mean(v) for i, v in per_item.items()}
        by_stratum = {}
        for i, q in item_quality.items():
            by_stratum.setdefault(items[i]["stratum"], []).append(q)
        secs = [a["seconds"] for a in answers]
        failure_rate = sum(_failed(r, MODELS[key]) for r in mine) / len(mine)
        out[key] = {
            "model": key,
            "tag": MODELS[key]["tag"],
            "n_items": len(item_quality),
            "n_calls": len(mine),
            "quality": round(statistics.mean(item_quality.values()), 4) if item_quality else 0.0,
            "quality_by_stratum": {s: round(statistics.mean(v), 3) for s, v in sorted(by_stratum.items())},
            "metrics": {f"{s}.{k}": round(statistics.mean(v), 3) for (s, k), v in sorted(metric_sums.items())},
            "item_quality": item_quality,
            "run_spread": round(statistics.mean(max(v) - min(v) for v in per_item.values()), 3) if per_item else None,
            "median_seconds": round(statistics.median(secs), 1) if secs else None,
            "p90_seconds": round(_pct(secs, 0.9), 1) if secs else None,
            "median_ttft": round(statistics.median([a["ttft_visible"] for a in answers if a["ttft_visible"] is not None] or [float("nan")]), 1),
            "median_tok_per_s": round(statistics.median([a["tok_per_s"] for a in answers if a["tok_per_s"]] or [float("nan")]), 1),
            "median_output_tokens": statistics.median([a["output_tokens"] for a in answers]) if answers else None,
            "thinking_share": round(sum(a["thinking_tokens"] for a in answers) / max(sum(a["output_tokens"] for a in answers), 1), 2),
            "failure_rate": round(failure_rate, 3),
            "json_valid": round(sum(json_valid) / len(json_valid), 3) if json_valid else None,
            "tiebreak_correct": round(sum(json_correct) / len(json_correct), 3) if json_correct else None,
            "leak_rate": round(leaks / len(answers), 3) if answers else 0.0,
            "vram_gb": fit.get(key, {}).get("vram_gb"),
        }
        o = out[key]
        o["gates"] = mb.gate_results(bool(fit.get(key, {}).get("fits")), failure_rate, o["json_valid"], o["leak_rate"])
    base = out.get(BASELINE)
    for key, o in out.items():
        if base and key != BASELINE:
            shared = sorted(set(o["item_quality"]) & set(base["item_quality"]))
            o["vs_baseline"] = {k: round(v, 3) for k, v in mb.paired_bootstrap(
                [o["item_quality"][i] for i in shared], [base["item_quality"][i] for i in shared]).items()}
    ranked = sorted(out.values(), key=lambda r: -r["quality"])
    pick = mb.recommend(ranked)
    SCORES.write_text(json.dumps({"models": out, "recommended": pick}, indent=1))

    print(f"\n{'model':20} {'qual':>6} {'vs base (95% CI)':>24} {'med s':>6} {'p90 s':>6} {'ttft':>5} {'tok/s':>6} {'think':>6} {'fail':>5} {'json':>5} {'leak':>5}  gates")
    for o in ranked:
        vb = o.get("vs_baseline")
        vb_s = f"{vb['diff']:+.3f} [{vb['lo']:+.3f},{vb['hi']:+.3f}]" if vb else "baseline"
        bad = [g for g, ok in o["gates"].items() if not ok]
        print(f"{o['model']:20} {o['quality']:6.3f} {vb_s:>24} {o['median_seconds']:6} {o['p90_seconds']:6} {o['median_ttft']:5} "
              f"{o['median_tok_per_s']:6} {o['thinking_share']:6} {o['failure_rate']:5} {str(o['json_valid']):>5} {o['leak_rate']:5}  "
              f"{'pass' if not bad else 'FAIL ' + ','.join(bad)}")
    print("\nquality by stratum:")
    for o in ranked:
        print(f"  {o['model']:20} {o['quality_by_stratum']}  run-to-run spread {o['run_spread']}")
    print(f"\nrecommended by the decision rule: {pick}")


# ---------------------------------------------------------------- blind review

def cmd_blind(args) -> None:
    items = load_items()
    calls = {}
    for c in load_calls():
        calls.setdefault(c["item"], []).append(c)
    chosen, seen = [], {}
    for it in items:
        n = seen.get(it["stratum"], 0)
        if n < BLIND_PER_STRATUM.get(it["stratum"], 0):
            chosen.append(it)
            seen[it["stratum"]] = n + 1
    rows = [r for r in read_jsonl(REPLAY) if r["run"] == 0 and (not args.models or r["model"] in args.models)]
    rng = random.Random(20261012)
    sheet, key = [], {}
    for it in chosen:
        seq = answer_seq(it, calls[it["id"]])
        answers = [r for r in rows if r["item"] == it["id"] and r["seq"] == seq]
        rng.shuffle(answers)
        sheet.append(f"\n\n{'=' * 100}\nITEM {it['id']} [{it['stratum']}]\nQUESTION: {it['message']}\nGOLD: {json.dumps(it.get('gold', {}), ensure_ascii=False)}")
        for r in answers:
            code = hashlib.sha1(f"{it['id']}|{r['model']}|{rng.random()}".encode()).hexdigest()[:6]
            key[code] = {"item": it["id"], "model": r["model"]}
            text = r["visible"] or "(no answer produced)"
            if len(text) > BLIND_MAX_CHARS:
                text = f"{text[:BLIND_MAX_CHARS]}\n[... {len(text) - BLIND_MAX_CHARS} more characters; ended: {r['done_reason']}]"
            sheet.append(f"\n{'-' * 60}\nANSWER {code}\n{text}")
    (OUT / "blind_sheet.txt").write_text("".join(sheet))
    (OUT / "blind_key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(key)} answers over {len(chosen)} items -> {OUT / 'blind_sheet.txt'} (key kept in blind_key.json)")


def cmd_unblind(args) -> None:
    """blind_scores.json maps answer code -> {correct, complete, clear, safe} each 0-2."""
    key = json.loads((OUT / "blind_key.json").read_text())
    scores = json.loads((OUT / "blind_scores.json").read_text())
    per = {}
    for code, s in scores.items():
        per.setdefault(key[code]["model"], []).append(s)
    print(f"{'model':20} {'n':>3} {'correct':>8} {'complete':>9} {'clear':>6} {'safe':>5} {'total/8':>8}")
    for model, ss in sorted(per.items(), key=lambda kv: -statistics.mean(sum(s.values()) for s in kv[1])):
        cols = [statistics.mean(s[k] for s in ss) for k in ("correct", "complete", "clear", "safe")]
        print(f"{model:20} {len(ss):3} {cols[0]:8.2f} {cols[1]:9.2f} {cols[2]:6.2f} {cols[3]:5.2f} {sum(cols):8.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["fit", "capture", "replay", "score", "blind", "unblind"])
    p.add_argument("--models", type=lambda s: s.split(","))
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--screen", action="store_true", help="replay only the fixed screening subset")
    args = p.parse_args()
    for m in args.models or []:
        if m not in MODELS:
            p.error(f"unknown model {m}; known: {', '.join(MODELS)}")
    if args.command == "capture":
        asyncio.run(_capture())
    else:
        {"fit": cmd_fit, "replay": cmd_replay, "score": cmd_score, "blind": cmd_blind, "unblind": cmd_unblind}[args.command](args)


if __name__ == "__main__":
    main()
