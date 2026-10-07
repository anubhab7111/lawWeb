import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import intent_classifier as ic

LABELS = ["crime_report", "document_analysis", "find_lawyer", "general_query", "non_legal"]


def _fake_head(monkeypatch, logits_for_doc_flag=0.0):
    dim = 4
    W = np.zeros((len(LABELS), dim + 1), dtype=np.float32)
    W[0, 0] = W[2, 1] = W[3, 2] = W[4, 3] = 5.0
    W[1, dim] = logits_for_doc_flag
    monkeypatch.setattr(ic, "_load_intent_head", lambda: (W, np.zeros(len(LABELS), dtype=np.float32), LABELS))


def test_clear_message_is_confident_and_unambiguous(monkeypatch):
    _fake_head(monkeypatch)
    result = ic._classify_with_head([1.0, 0.0, 0.0, 0.0], has_document=False)
    assert result.primary_intent == "crime_report"
    assert not result.is_ambiguous and result.secondary_intents == []
    assert abs(sum(result.scores.values()) - 1.0) < 1e-6


def test_message_split_three_ways_is_ambiguous_with_all_contenders(monkeypatch):
    _fake_head(monkeypatch)
    result = ic._classify_with_head([0.6, 0.0, 0.6, 0.6], has_document=False)
    assert result.is_ambiguous
    assert {result.primary_intent, *result.secondary_intents} == {"crime_report", "general_query", "non_legal"}


def test_unsure_non_legal_is_answered_as_a_legal_question(monkeypatch):
    _fake_head(monkeypatch)
    sure = ic._classify_with_head([0.0, 0.0, 0.0, 1.0], has_document=False)
    unsure = ic._classify_with_head([0.4, 0.0, 0.4, 0.5], has_document=False)
    assert sure.primary_intent == "non_legal"
    assert unsure.scores["non_legal"] < ic.NON_LEGAL_MIN_PROB
    assert unsure.primary_intent == "general_query" and not unsure.is_ambiguous


def test_attached_document_feeds_the_document_feature(monkeypatch):
    _fake_head(monkeypatch, logits_for_doc_flag=12.0)
    q = [0.0, 0.0, 1.0, 0.0]
    assert ic._classify_with_head(q, has_document=False).primary_intent == "general_query"
    assert ic._classify_with_head(q, has_document=True).primary_intent == "document_analysis"
