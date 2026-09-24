#!/usr/bin/env python3
"""
train_lsi_classifier.py — fine-tune InLegalBERT for IL-TUR statute identification.

Train on the `lsi` train split, select the epoch on dev (official macro-F1 with a
dev-fitted threshold), then write per-case label probabilities for dev and test into
the score cache (<corpus>/builds/scores/classifier/...) for report_iltur.py.

Fits a 4GB GPU: fp16 autocast, gradient checkpointing, micro-batches bounded by a
chunk budget, gradient accumulation. Checkpoints go to <corpus>/builds/classifier/
and a killed run resumes from the last one.

Usage (from server/, Ollama idle):
    python train_lsi_classifier.py --bench --max-chunks 2        # measure throughput
    python train_lsi_classifier.py --max-chunks 2 --epochs 3
    python train_lsi_classifier.py --score-only                  # write dev/test scores from best.pt
"""

import argparse
import math
import os
import pickle
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from app.ingest.iltur_export import lsi_dir  # noqa: E402
from app.ingest.paths import corpus_path  # noqa: E402
from app.ingest.thermal import ThermalGuard  # noqa: E402
from app.metrics import iltur_official as io  # noqa: E402
from app.metrics.iltur_loader import label_names  # noqa: E402
from app.tools import statute_classifier as sc  # noqa: E402

N_LABELS = 100


def work_dir() -> Path:
    d = corpus_path("builds", "classifier")
    d.mkdir(parents=True, exist_ok=True)
    return d


def tokenized(split: str, tokenizer, max_chunks: int):
    """[(case id, chunks, label ids)] for a split, cached on the drive."""
    path = work_dir() / f"tokens_{split}_c{max_chunks}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    df = pd.read_parquet(lsi_dir() / f"{split}.parquet", columns=["id", "sentences", "label_ids"])
    rows = []
    for i, r in enumerate(df.itertuples(), start=1):
        rows.append((str(r.id), sc.chunk_ids(tokenizer, sc.clean_facts(list(r.sentences)), max_chunks), list(r.label_ids)))
        if i % 5000 == 0:
            print(f"  tokenized {split} {i}/{len(df)}", flush=True)
    with open(path, "wb") as f:
        pickle.dump(rows, f)
    return rows


def micro_batches(rows, chunk_budget: int, shuffle: bool, seed: int = 0):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batch, used = [], 0
    for i in order:
        n = len(rows[i][1])
        if batch and used + n > chunk_budget:
            yield batch
            batch, used = [], 0
        batch.append(i)
        used += n
    if batch:
        yield batch


def targets(rows, idx):
    y = torch.zeros(len(idx), N_LABELS)
    for j, i in enumerate(idx):
        y[j, rows[i][2]] = 1.0
    return y


def pos_weight(rows, cap: float):
    counts = np.zeros(N_LABELS)
    for _, _, labels in rows:
        counts[labels] += 1
    neg = len(rows) - counts
    w = np.sqrt(neg / np.maximum(counts, 1))
    return torch.tensor(np.minimum(w, cap), dtype=torch.float)


@torch.no_grad()
def predict(model, rows, pad_id, chunk_budget, device):
    model.eval()
    probs = np.zeros((len(rows), N_LABELS), dtype=np.float32)
    for idx in micro_batches(rows, chunk_budget * 2, shuffle=False):
        ids, mask, owner, n = sc.collate([rows[i][1] for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(ids.to(device), mask.to(device), owner.to(device), n)
        probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    model.train()
    return probs


def dev_macro_f1(probs, rows):
    gold = np.zeros_like(probs, dtype=np.int8)
    for j, (_, _, labels) in enumerate(rows):
        gold[j, labels] = 1
    t = io.fit_global_threshold(probs, gold, grid=np.linspace(0.05, 0.95, 37))
    return io.official_macro_f1(gold, io.predict_threshold(probs, [t] * N_LABELS)), t


def write_scores(rows, probs, split):
    names = label_names()
    out = corpus_path("builds", "scores", "classifier", split)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.pkl"):
        old.unlink()
    scores = {cid: {names[j]: float(p) for j, p in enumerate(probs[i]) if p >= 1e-4} for i, (cid, _, _) in enumerate(rows)}
    with open(out / "shard_0000.pkl", "wb") as f:
        pickle.dump(scores, f)


def build_model(device, grad_ckpt=True):
    encoder = AutoModel.from_pretrained(sc.BASE_MODEL)
    if grad_ckpt:
        encoder.gradient_checkpointing_enable()
    return sc.ChunkedStatuteClassifier(encoder, N_LABELS).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--max-chunks", type=int, default=2)
    parser.add_argument("--chunk-budget", type=int, default=4, help="max chunks per micro-batch (GPU memory)")
    parser.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"c{args.max_chunks}"

    tokenizer = AutoTokenizer.from_pretrained(sc.BASE_MODEL)
    pad = tokenizer.pad_token_id
    train = tokenized("train", tokenizer, args.max_chunks)
    dev = tokenized("dev", tokenizer, args.max_chunks)
    model = build_model(device)
    ckpt_dir = work_dir() / tag
    ckpt_dir.mkdir(exist_ok=True)

    if args.score_only:
        model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
        test = tokenized("test", tokenizer, args.max_chunks)
        for split, rows in (("dev", dev), ("test", test)):
            probs = predict(model, rows, pad, args.chunk_budget, device)
            write_scores(rows, probs, split)
            print(f"[classifier] wrote {split} scores ({len(rows)} cases)")
        return

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    optim = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    batches_per_epoch = sum(1 for _ in micro_batches(train, args.chunk_budget, shuffle=False))
    total_steps = math.ceil(batches_per_epoch / args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(train, args.pos_weight_cap).to(device))

    state = {"epoch": 0, "micro": 0, "best": -1.0}
    last = ckpt_dir / "last.pt"
    if last.exists() and not args.bench:
        saved = torch.load(last, map_location=device)
        model.load_state_dict(saved["model"])
        optim.load_state_dict(saved["optim"])
        sched.load_state_dict(saved["sched"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        print(f"[classifier] resumed at epoch {state['epoch']} micro-batch {state['micro']}")

    model.train()
    guard = ThermalGuard()
    started = time.time()
    for epoch in range(state["epoch"], args.epochs):
        seen_docs = 0
        for m, idx in enumerate(micro_batches(train, args.chunk_budget, shuffle=True, seed=epoch)):
            if m < state["micro"]:
                continue
            ids, mask, owner, n = sc.collate([train[i][1] for i in idx], pad)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(ids.to(device), mask.to(device), owner.to(device), n)
            loss = loss_fn(logits.float(), targets(train, idx).to(device)) / args.accum
            scaler.scale(loss).backward()
            guard.step()
            seen_docs += len(idx)
            if (m + 1) % args.accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
            if args.bench and m == 60:
                rate = seen_docs / (time.time() - started)
                print(f"[bench] max_chunks={args.max_chunks}: {rate:.2f} docs/s -> {len(train) / rate / 3600:.2f} h/epoch; "
                      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
                return
            if (m + 1) % 2000 == 0:
                state["micro"] = m + 1
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "state": state}, last)
                print(f"[classifier] epoch {epoch + 1} micro {m + 1}/{batches_per_epoch} loss {loss.item() * args.accum:.4f} "
                      f"({time.time() - started:.0f}s)", flush=True)
        probs = predict(model, dev, pad, args.chunk_budget, device)
        f1, t = dev_macro_f1(probs, dev)
        print(f"[classifier] epoch {epoch + 1}: dev macro-F1 {f1:.2f} (threshold {t:.2f})", flush=True)
        if f1 > state["best"]:
            state["best"] = f1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        state["epoch"], state["micro"] = epoch + 1, 0
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, last)
    print(f"[classifier] best dev macro-F1 {state['best']:.2f}; run --score-only to write scores")


if __name__ == "__main__":
    main()
