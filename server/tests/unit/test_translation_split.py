import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.multilingual.translation import split_for_translation


def test_short_line_untouched():
    assert split_for_translation("Section 302 IPC.") == ["Section 302 IPC."]


def test_long_text_is_fully_covered_in_bounded_pieces():
    text = " ".join(f"Sentence number {i} explains the rule in detail." for i in range(200))
    pieces = split_for_translation(text)
    assert all(len(p) <= 400 for p in pieces)
    assert " ".join(pieces).split() == text.split()


def test_unbroken_long_sentence_is_split_on_words():
    text = "word " * 500
    pieces = split_for_translation(text.strip())
    assert all(len(p) <= 400 for p in pieces)
    assert " ".join(pieces).split() == text.split()
