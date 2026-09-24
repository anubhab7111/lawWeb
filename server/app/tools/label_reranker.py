"""
label_reranker.py — cross-encoder reranking of candidate statutes against case facts.

Candidate statutes come from every retrieval source (precedent votes, the
classifier, judgment votes, statute retrieval). The cross-encoder then reads the
facts *together with each candidate's statute text* and scores the pair, so the
final order reflects whether the facts actually satisfy the provision rather than
how the candidate was found. Facts longer than one input are read as several
segments and a candidate keeps its best segment score (the offence may be described
anywhere in the narrative).

The model is a cross-encoder fine-tuned on IL-TUR training cases
(train_label_reranker.py); statute texts ship as a fixture (app/data/iltur_statutes.json)
so scoring never needs the corpus drive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

STATUTES_PATH = Path(__file__).resolve().parent.parent / "data" / "iltur_statutes.json"
BASE_MODEL = "BAAI/bge-reranker-base"
FACT_TOKENS = 380
STATUTE_TOKENS = 120

_statutes: Optional[Dict[str, str]] = None


def statutes() -> Dict[str, str]:
    global _statutes
    if _statutes is None:
        _statutes = json.loads(STATUTES_PATH.read_text())
    return _statutes


def statute_text(label: str) -> str:
    return f"{label}. {statutes().get(label, '')}"


def fact_segments(tokenizer, text: str, max_segments: int, seg_tokens: int = FACT_TOKENS) -> List[str]:
    """Up to max_segments token windows of the facts, spread evenly across them."""
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    starts = list(range(0, max(len(ids), 1), seg_tokens))
    if len(starts) > max_segments:
        starts = [starts[round(i * (len(starts) - 1) / max(max_segments - 1, 1))] for i in range(max_segments)]
    return [tokenizer.decode(ids[s : s + seg_tokens]) for s in starts] or [text]


def encode_pairs(tokenizer, facts: Sequence[str], labels: Sequence[str]):
    return tokenizer(
        list(facts),
        [statute_text(l) for l in labels],
        truncation="longest_first",
        max_length=FACT_TOKENS + STATUTE_TOKENS + 4,
        padding=True,
        return_tensors="pt",
    )


class LabelReranker:
    def __init__(self, model_dir: str, device: str = "cuda", max_segments: int = 3, batch_size: int = 64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        if device == "cuda":
            self.model.half()
        self.device = device
        self.max_segments = max_segments
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, facts_text: str, candidates: Sequence[str]) -> Dict[str, float]:
        """Probability-like score per candidate label: max over fact segments."""
        if not candidates:
            return {}
        segments = fact_segments(self.tokenizer, facts_text, self.max_segments)
        pairs = [(seg, label) for seg in segments for label in candidates]
        logits = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            enc = encode_pairs(self.tokenizer, [p[0] for p in chunk], [p[1] for p in chunk]).to(self.device)
            logits.append(self.model(**enc).logits.float().squeeze(-1).cpu().numpy())
        scores = np.concatenate(logits).reshape(len(segments), len(candidates)).max(axis=0)
        return {label: float(1 / (1 + np.exp(-s))) for label, s in zip(candidates, scores)}
