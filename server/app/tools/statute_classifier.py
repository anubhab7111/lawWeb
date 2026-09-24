"""
statute_classifier.py — multi-label statute identification from case facts.

A legal-domain BERT (InLegalBERT) reads the facts in 512-token chunks; the chunk
[CLS] vectors are combined by learned attention pooling (so the passages that
actually carry the offence dominate) and a linear layer scores the 100 IL-TUR
statute labels independently (sigmoid). This is the model family the IL-TUR paper
reports for `lsi` (InLegalBERT 26.23 macro-F1); the chunking is what lets it see
more than the first 512 tokens of a ~2k-token narrative.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import torch
from torch import nn

BASE_MODEL = "law-ai/InLegalBERT"
_ENTITY = re.compile(r"<ENTITY>")


def clean_facts(sentences: Sequence[str]) -> str:
    return " ".join(_ENTITY.sub("[UNK]", s).strip() for s in sentences if s.strip())


def chunk_ids(tokenizer, text: str, max_chunks: int, chunk_len: int = 512, stride: int = 0) -> List[List[int]]:
    """Token-id chunks of at most chunk_len (incl. [CLS]/[SEP]); when a document has
    more chunks than max_chunks, chunks are sampled evenly across it."""
    body = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    width = chunk_len - 2
    step = width - stride
    starts = list(range(0, max(len(body) - stride, 1), step)) or [0]
    if len(starts) > max_chunks:
        pick = [round(i * (len(starts) - 1) / (max_chunks - 1)) for i in range(max_chunks)] if max_chunks > 1 else [0]
        starts = [starts[i] for i in pick]
    return [[tokenizer.cls_token_id] + body[s : s + width] + [tokenizer.sep_token_id] for s in starts]


class ChunkedStatuteClassifier(nn.Module):
    def __init__(self, encoder, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask, doc_index, n_docs):
        """input_ids: (total_chunks, L); doc_index: (total_chunks,) owning document."""
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits_a = self.attn(cls).squeeze(-1)
        pooled = cls.new_zeros(n_docs, cls.size(-1))
        for d in range(n_docs):
            sel = doc_index == d
            w = torch.softmax(logits_a[sel], dim=0).unsqueeze(-1)
            pooled[d] = (w * cls[sel]).sum(0)
        return self.head(self.drop(pooled))


SERVING_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "faiss_index" / "statute_classifier"


class StatuteClassifier:
    """Serving wrapper: facts -> {label: probability}, plus per-label thresholds
    fitted on IL-TUR dev (the decision rule behind the 38.35 test macro-F1)."""

    def __init__(self, model_dir=SERVING_DIR, device: str = "cpu", max_chunks: int = 2):
        import json

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        meta = json.loads((model_dir / "meta.json").read_text())
        self.labels: List[str] = meta["labels"]
        self.thresholds = dict(zip(self.labels, meta["thresholds"]))
        self.tokenizer = AutoTokenizer.from_pretrained(meta["base_model"])
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(meta["base_model"]))
        self.model = ChunkedStatuteClassifier(encoder, len(self.labels))
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.to(device).eval()
        self.device, self.max_chunks = device, max_chunks

    @property
    def available(self) -> bool:
        return True

    @torch.no_grad()
    def predict(self, sentences: Sequence[str]) -> dict:
        chunks = chunk_ids(self.tokenizer, clean_facts(sentences), self.max_chunks)
        ids, mask, owner, n = collate([chunks], self.tokenizer.pad_token_id)
        logits = self.model(ids.to(self.device), mask.to(self.device), owner.to(self.device), n)
        probs = torch.sigmoid(logits.float())[0].cpu().tolist()
        return dict(zip(self.labels, probs))

    def decide(self, probs: dict, max_labels: int = 6) -> List[tuple]:
        """Labels clearing their dev-fitted threshold, best first (at least one)."""
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        chosen = [(l, p) for l, p in ranked if p >= self.thresholds.get(l, 0.5)]
        return (chosen or ranked[:1])[:max_labels]


def collate(batch_chunks: Sequence[List[List[int]]], pad_id: int):
    """Flatten per-document chunk lists into padded tensors + owner index."""
    flat, owner = [], []
    for d, chunks in enumerate(batch_chunks):
        flat.extend(chunks)
        owner.extend([d] * len(chunks))
    length = max(len(c) for c in flat)
    ids = torch.full((len(flat), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(flat), length), dtype=torch.long)
    for i, c in enumerate(flat):
        ids[i, : len(c)] = torch.tensor(c)
        mask[i, : len(c)] = 1
    return ids, mask, torch.tensor(owner), len(batch_chunks)
