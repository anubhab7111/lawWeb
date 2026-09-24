import torch
from transformers import BertConfig, BertModel

from app.tools import statute_classifier as sc


class FakeTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": [1000 + i for i, _ in enumerate(text.split())]}


def test_clean_facts_masks_entities():
    assert sc.clean_facts(["<ENTITY> killed <ENTITY>.", " "]) == "[UNK] killed [UNK]."


def test_chunks_are_bounded_wrapped_and_evenly_sampled():
    tok = FakeTokenizer()
    text = " ".join(["w"] * 100)
    chunks = sc.chunk_ids(tok, text, max_chunks=3, chunk_len=12)
    assert len(chunks) == 3
    assert all(c[0] == 101 and c[-1] == 102 and len(c) <= 12 for c in chunks)
    assert chunks[0][1] == 1000 and chunks[-1][-2] == 1099  # first and last tokens kept
    assert sc.chunk_ids(tok, "a b", max_chunks=4, chunk_len=12) == [[101, 1000, 1001, 102]]


def test_model_pools_chunks_per_document():
    cfg = BertConfig(vocab_size=2000, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    model = sc.ChunkedStatuteClassifier(BertModel(cfg), n_labels=5).eval()
    ids, mask, owner, n = sc.collate([[[101, 5, 102]], [[101, 6, 7, 102], [101, 8, 102]]], pad_id=0)
    assert ids.shape == (3, 4) and owner.tolist() == [0, 1, 1] and n == 2
    with torch.no_grad():
        logits = model(ids, mask, owner, n)
    assert logits.shape == (2, 5)
