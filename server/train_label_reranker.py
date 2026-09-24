#!/usr/bin/env python3
"""
train_label_reranker.py — fine-tune the statute cross-encoder (app/tools/label_reranker.py).

Training pairs come from IL-TUR train cases: (fact segment, statute text) with label
1 for the case's gold statutes and 0 for hard negatives — the non-gold statutes the
precedent index proposes for that case (score_iltur.py precedent --split train,
self-masked) — plus a few random statutes. Evaluation reranks the candidates
precedent proposes for dev cases (gold is never injected) and reports how much the
reranked order improves over precedent's own.

Fits a 4GB GPU: fp16 autocast, gradient checkpointing, gradient accumulation, and
the 250k-token word-embedding matrix frozen (its gradients and AdamW state would
not fit). Checkpoints every --ckpt-every micro-batches; a killed run resumes.
The model is saved to <corpus>/builds/reranker/<tag>/.

Usage (from server/, Ollama idle):
    python train_label_reranker.py --bench
    python train_label_reranker.py --cases 20000 --tag v1
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from dotenv import load_dotenv

load_dotenv(_SERVER_DIR / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from app.ingest.iltur_export import lsi_dir  # noqa: E402
from app.ingest.paths import corpus_path  # noqa: E402
from app.ingest.thermal import ThermalGuard  # noqa: E402
from app.metrics.iltur_loader import label_names  # noqa: E402
from app.tools import label_reranker as lr  # noqa: E402
from app.tools.statute_classifier import clean_facts  # noqa: E402
from score_iltur import load_scores  # noqa: E402


def top_candidates(scores, k):
    return [l for l, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


def build_pairs(tokenizer, split, n_cases, hard_k, n_random, seed, segments=2):
    names = list(label_names())
    df = pd.read_parquet(lsi_dir() / f"{split}.parquet", columns=["id", "sentences", "label_ids"])
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    idx = idx[:n_cases]
    prec = load_scores("precedent", split)
    pairs = []
    for i in idx:
        r = df.iloc[i]
        gold = {names[j] for j in r.label_ids}
        segs = lr.fact_segments(tokenizer, clean_facts(list(r.sentences)), segments)
        hard = [l for l in top_candidates(prec.get(str(r.id), {}), hard_k) if l not in gold]
        rand = rng.sample([l for l in names if l not in gold and l not in hard], n_random)
        for label in gold:
            pairs.append((rng.choice(segs), label, 1.0))
        for label in hard + rand:
            pairs.append((rng.choice(segs), label, 0.0))
    rng.shuffle(pairs)
    return pairs


@torch.no_grad()
def evaluate(model, tokenizer, device, n_cases=600, k=15, seed=7):
    """Rerank precedent's top-k dev candidates; compare MRR and hit@3 with precedent's order."""
    names = list(label_names())
    df = pd.read_parquet(lsi_dir() / "dev.parquet", columns=["id", "sentences", "label_ids"]).sample(n_cases, random_state=seed)
    prec = load_scores("precedent", "dev")
    model.eval()
    stats = {"prec_mrr": 0.0, "rr_mrr": 0.0, "prec_hit3": 0.0, "rr_hit3": 0.0}
    for r in df.itertuples():
        gold = {names[j] for j in r.label_ids}
        cands = top_candidates(prec.get(str(r.id), {}), k)
        if not cands:
            continue
        segs = lr.fact_segments(tokenizer, clean_facts(list(r.sentences)), 3)
        pairs = [(s, c) for s in segs for c in cands]
        enc = lr.encode_pairs(tokenizer, [p[0] for p in pairs], [p[1] for p in pairs]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**enc).logits.float().squeeze(-1).cpu().numpy()
        best = logits.reshape(len(segs), len(cands)).max(axis=0)
        reranked = [cands[j] for j in np.argsort(-best)]
        for name, order in (("prec", cands), ("rr", reranked)):
            rank = next((i for i, c in enumerate(order, 1) if c in gold), None)
            stats[f"{name}_mrr"] += 1 / rank if rank else 0.0
            stats[f"{name}_hit3"] += float(bool(gold & set(order[:3])))
    model.train()
    return {k2: v / len(df) for k2, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--base", default=lr.BASE_MODEL)
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--hard-k", type=int, default=12)
    parser.add_argument("--random", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)
    model.gradient_checkpointing_enable()
    model.get_input_embeddings().weight.requires_grad_(False)
    out = corpus_path("builds", "reranker", args.tag)
    ckpt = corpus_path("builds", "reranker", f"{args.tag}.ckpt.pt")

    base_eval = evaluate(model, tokenizer, device, n_cases=150 if args.bench else 600)
    print(f"[reranker] before fine-tuning (dev): {base_eval}", flush=True)

    pairs = build_pairs(tokenizer, "train", args.cases if not args.bench else 400, args.hard_k, args.random, seed=0)
    pos = sum(p[2] for p in pairs)
    print(f"[reranker] {len(pairs)} training pairs ({pos:.0f} positive)", flush=True)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = len(pairs) // (args.micro * args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * steps), max(steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(pairs) - pos) / max(pos, 1)).sqrt().to(device))

    state = {"epoch": 0, "micro": 0}
    if ckpt.exists() and not args.bench:
        saved = torch.load(ckpt, map_location=device)
        model.load_state_dict(saved["model"])
        optim.load_state_dict(saved["optim"])
        sched.load_state_dict(saved["sched"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        print(f"[reranker] resumed at epoch {state['epoch'] + 1} micro-batch {state['micro']}", flush=True)

    model.train()
    guard = ThermalGuard()
    started = time.time()
    for epoch in range(state["epoch"], args.epochs):
        for m, i in enumerate(range(0, len(pairs), args.micro)):
            if m < state["micro"]:
                continue
            batch = pairs[i : i + args.micro]
            enc = lr.encode_pairs(tokenizer, [b[0] for b in batch], [b[1] for b in batch]).to(device)
            y = torch.tensor([b[2] for b in batch], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits.float().squeeze(-1)
            scaler.scale(loss_fn(logits, y) / args.accum).backward()
            guard.step()
            if (m + 1) % args.accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
            if args.bench and m == 100:
                rate = (m + 1) * args.micro / (time.time() - started)
                print(f"[bench] {rate:.1f} pairs/s; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; "
                      f"{args.cases} cases (~{args.cases * 16} pairs) -> {args.cases * 16 / rate / 3600:.1f} h/epoch")
                return
            if (m + 1) % args.ckpt_every == 0 and (m + 1) % args.accum == 0:
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "state": {"epoch": epoch, "micro": m + 1}}, ckpt)
                print(f"[reranker] epoch {epoch + 1} {i + args.micro}/{len(pairs)} pairs ({time.time() - started:.0f}s)", flush=True)
        state["micro"] = 0
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": {"epoch": epoch + 1, "micro": 0}}, ckpt)
        print(f"[reranker] epoch {epoch + 1} (dev): {evaluate(model, tokenizer, device)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[reranker] saved {out}")


if __name__ == "__main__":
    main()
