import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools import crime_reporter


def _fake_classifier(monkeypatch, labels):
    W = np.eye(len(labels), 4, dtype=np.float32)
    b = np.zeros(len(labels), dtype=np.float32)
    monkeypatch.setattr(crime_reporter, "_load_classifier", lambda: (W, b, labels))


def test_merged_family_resolved_by_keywords():
    assert crime_reporter.resolve_family("Theft or Robbery", "they robbed me at knifepoint") == "robbery"
    assert crime_reporter.resolve_family("Theft or Robbery", "something went missing") == "theft"


def test_general_families_map_to_general():
    assert crime_reporter.resolve_family("Narcotics", "ganja was found") == "general"


def test_vector_picks_highest_scoring_family(monkeypatch):
    _fake_classifier(monkeypatch, ["Cyber Crime", "Murder", "Kidnapping", "Others"])
    q = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)
    assert crime_reporter.crime_type_from_vector(q, "anything") == "murder"


def test_falls_back_to_keywords_without_weights(monkeypatch):
    monkeypatch.setattr(crime_reporter, "_load_classifier", lambda: None)
    assert asyncio.run(crime_reporter.classify_crime_type("my phone was stolen")) == "theft"


def test_falls_back_to_keywords_when_embedding_fails(monkeypatch):
    _fake_classifier(monkeypatch, ["Cyber Crime", "Murder", "Kidnapping", "Others"])

    async def broken():
        raise RuntimeError("no model")

    monkeypatch.setattr("app.tools.base_legal_rag._get_shared_embeddings", broken)
    assert asyncio.run(crime_reporter.classify_crime_type("my phone was stolen")) == "theft"
