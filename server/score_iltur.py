#!/usr/bin/env python3
"""
score_iltur.py — per-case label scores for a whole IL-TUR `lsi` split, cached.

Each system maps a case's facts to {label name: score} over the official 100-label
vocabulary (exact names: "Section 294(b)" is not "Section 294"). Scores for every
case of a split are written in resumable shards to
<corpus>/builds/scores/<system>/<split>/shard_NNNN.pkl, so decision rules and fusion
(`report_iltur.py`) can be tuned in seconds instead of re-running retrieval.

Systems:
  precedent   similar past IL-TUR train+dev cases vote for their labels
              (dev cases are masked out of the index when scoring dev)

  judgments   Supreme Court passages similar to the facts vote for the sections
              their judgments cite; judgments overlapping the scored split's IL-TUR
              cases (decontaminate.py) are masked here, never removed from the index
  reranker    the fine-tuned statute cross-encoder rescores the top candidates
              proposed by precedent and the classifier (needs their scores)

Robustness variants of precedent (written as their own systems):
  --memory train        mask every dev case: train-only memory, like the paper's
                        baselines, which learned from the train split alone
  --exclude-overlap F   also mask, per case, every indexed case sharing at least a
                        fraction F of its 10-word shingles (near-duplicate facts)

Usage (from server/):
    EMBEDDINGS_DEVICE=cuda python score_iltur.py precedent --split dev
    EMBEDDINGS_DEVICE=cuda python score_iltur.py precedent --split test
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

_SERVER_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SERVER_DIR))
os.chdir(_SERVER_DIR)

from dotenv import load_dotenv

load_dotenv(_SERVER_DIR / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ingest.iltur_export import lsi_dir  # noqa: E402
from app.ingest.paths import corpus_path  # noqa: E402
from app.metrics.iltur_eval import fact_windows  # noqa: E402
from app.metrics.iltur_loader import label_names  # noqa: E402
from app.tools import precedent_rag as pr  # noqa: E402

SHARD = 512


def shard_dir(system: str, split: str) -> Path:
    return corpus_path("builds", "scores", system, split)


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(lsi_dir() / f"{split}.parquet", columns=["id", "sentences", "label_ids"])
    df["id"] = df["id"].astype(str)
    return df


def load_scores(system: str, split: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for path in sorted(shard_dir(system, split).glob("shard_*.pkl")):
        with open(path, "rb") as f:
            out.update(pickle.load(f))
    return out


def gold_labels(split: str) -> Dict[str, List[str]]:
    names = label_names()
    df = pd.read_parquet(lsi_dir() / f"{split}.parquet", columns=["id", "label_ids"])
    return {str(r.id): [names[i] for i in r.label_ids] for r in df.itertuples()}


class PrecedentScorer:
    """Batched precedent voting over exact label names."""

    def __init__(self, split: str, device: str, memory: str = "all", overlap=None):
        from app.tools.base_legal_rag import _make_bge_embeddings

        self.index = pr.get_precedent_index()
        self.index.load()
        names = label_names()
        id_to_names = {}
        for s in ("train", "dev"):
            for r in pd.read_parquet(lsi_dir() / f"{s}.parquet", columns=["id", "label_ids"]).itertuples():
                id_to_names[str(r.id)] = [names[i] for i in r.label_ids]
        # label names in the index's own case order (its meta stores bare numbers)
        self.case_labels = [id_to_names[cid] for cid in self.index.case_ids]
        self.lookup = {cid: i for i, cid in enumerate(self.index.case_ids)}
        self.mask_self = split != "test"
        self.always_masked = set()
        if memory == "train":
            dev_ids = pd.read_parquet(lsi_dir() / "dev.parquet", columns=["id"])["id"].astype(str)
            self.always_masked = {self.lookup[c] for c in dev_ids if c in self.lookup}
        # per-case masks: case id -> indexed case ids sharing too much text
        self.overlap = {cid: {self.lookup[o] for o in others if o in self.lookup} for cid, others in (overlap or {}).items()}
        emb = _make_bge_embeddings(device)
        emb.client.max_seq_length = 256
        if device == "cuda":
            emb.client.half()
        self.emb = emb

    def score(self, rows: pd.DataFrame, k: int = pr.NEIGHBORS, power: float = pr.VOTE_POWER) -> Dict[str, Dict[str, float]]:
        windows, owner = [], []
        for i, sentences in enumerate(rows["sentences"]):
            w = fact_windows(list(sentences), window=4, stride=3, max_windows=pr.QUERY_WINDOWS)
            windows.extend(w)
            owner.extend([i] * len(w))
        vecs = pr.encode_texts(self.emb, windows, batch_size=128)
        extra = (k if self.mask_self else 0) + (3 * k if self.always_masked or self.overlap else 0)
        sims, cases = self.index.search(vecs, k, extra=extra)
        owner = np.asarray(owner)
        out = {}
        for i, cid in enumerate(rows["id"]):
            sel = owner == i
            exclude = set(self.always_masked) | self.overlap.get(cid, set())
            if self.mask_self and cid in self.lookup:
                exclude.add(self.lookup[cid])
            out[cid] = pr.vote_labels(sims[sel], cases[sel], self.case_labels, exclude or None, power) if sel.any() else {}
        return out


class JudgmentScorer:
    """Judgment-passage voting, with IL-TUR-overlapping judgments masked."""

    def __init__(self, split: str, device: str, k: int = 50, power: float = 8.0):
        from app.tools.base_legal_rag import _make_bge_embeddings
        from app.tools.judgment_rag import get_judgment_index

        self.index = get_judgment_index()
        self.index.load()
        report = corpus_path("builds", "decontam", "sc_flagged.json" if split == "test" else f"sc_flagged_{split}.json")
        self.masked = set(json.loads(report.read_text())["flagged_docs"]) if report.exists() else set()
        if split in ("test", "dev") and not report.exists():
            raise SystemExit(f"{report} missing: run app.ingest.decontaminate --split {split} first")
        self.vocab = set(label_names())
        self.k, self.power = k, power
        emb = _make_bge_embeddings(device)
        emb.client.max_seq_length = 256
        if device == "cuda":
            emb.client.half()
        self.emb = emb

    def score(self, rows: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        out = {}
        for cid, sentences in zip(rows["id"], rows["sentences"]):
            windows = fact_windows(list(sentences), window=4, stride=3, max_windows=pr.QUERY_WINDOWS)
            if not windows:
                out[cid] = {}
                continue
            votes = self.index.section_votes(
                pr.encode_texts(self.emb, windows, batch_size=64), k=self.k, power=self.power,
                normalize=False, exclude_docs=self.masked,
            )
            out[cid] = {f"Section {n}": v for n, v in votes.items() if f"Section {n}" in self.vocab}
        return out


class RerankerScorer:
    """Fine-tuned cross-encoder over the top candidates of other systems."""

    def __init__(self, split: str, device: str, model: str, sources=("precedent", "classifier"), top: int = 12):
        from app.tools.label_reranker import LabelReranker
        from app.tools.statute_classifier import clean_facts

        self.reranker = LabelReranker(str(model), device=device)
        self.clean = clean_facts
        self.sources = [load_scores(src, split) for src in sources]
        self.top = top

    def candidates(self, cid: str) -> List[str]:
        out: List[str] = []
        for scores in self.sources:
            for label, _ in sorted(scores.get(cid, {}).items(), key=lambda kv: -kv[1])[: self.top]:
                if label not in out:
                    out.append(label)
        return out

    def score(self, rows: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        return {
            cid: self.reranker.score(self.clean(list(sentences)), self.candidates(cid))
            for cid, sentences in zip(rows["id"], rows["sentences"])
        }


def overlap_map(split: str, fraction: float) -> Dict[str, List[str]]:
    """case id -> train/dev case ids sharing >= fraction of its 10-word shingles (cached)."""
    from app.ingest.decontaminate import TestShingles, contaminated_cases, tokenize

    path = corpus_path("builds", "scores", f"overlap_{split}_{fraction}.pkl")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    memory = pd.concat([pd.read_parquet(lsi_dir() / f"{s}.parquet", columns=["id", "sentences"]) for s in ("train", "dev")])
    index = TestShingles.from_texts({str(r.id): " ".join(r.sentences) for r in memory.itertuples()})
    out = {}
    for r in load_split(split).itertuples():
        toks = tokenize(" ".join(r.sentences))
        need = max(1, int(fraction * max(len(toks) - 9, 1)))
        hits = contaminated_cases(index, toks, min_shingles=need)
        hits.pop(str(r.id), None)
        if hits:
            out[str(r.id)] = list(hits)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("system", choices=["precedent", "judgments", "reranker"])
    parser.add_argument("--reranker-model", help="reranker: fine-tuned model dir")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="first N cases only (smoke test)")
    parser.add_argument("--memory", choices=["all", "train"], default="all")
    parser.add_argument("--exclude-overlap", type=float)
    args = parser.parse_args()
    system = args.system
    if args.memory == "train":
        system += "_trainmem"
    if args.exclude_overlap:
        system += f"_strict{int(args.exclude_overlap * 100)}"

    df = load_split(args.split)
    if args.limit:
        df = df.head(args.limit)
    out_dir = shard_dir(system, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap = overlap_map(args.split, args.exclude_overlap) if args.exclude_overlap else None
    if overlap is not None:
        print(f"[{system}/{args.split}] {len(overlap)} cases have near-duplicate indexed cases (masked)", flush=True)
    if args.system == "judgments":
        scorer = JudgmentScorer(args.split, args.device)
    elif args.system == "reranker":
        scorer = RerankerScorer(args.split, args.device, args.reranker_model or corpus_path("builds", "reranker", "v1"))
    else:
        scorer = PrecedentScorer(args.split, args.device, memory=args.memory, overlap=overlap)

    started = time.time()
    n_shards = (len(df) + SHARD - 1) // SHARD
    for s in range(n_shards):
        path = out_dir / f"shard_{s:04d}.pkl"
        if path.exists():
            continue
        rows = df.iloc[s * SHARD : (s + 1) * SHARD]
        scores = scorer.score(rows)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(scores, f)
        tmp.rename(path)
        print(f"[{system}/{args.split}] shard {s + 1}/{n_shards} ({time.time() - started:.0f}s)", flush=True)
    print(f"[{system}/{args.split}] done: {len(load_scores(system, args.split))} cases")


if __name__ == "__main__":
    main()
